# ruff: noqa: E402
"""
Flexible, VRAM-safe comparison orchestrator for multiple evaluation variants.

Usage:
    python scripts/compare.py --spec configs/compare_gsm8k.yaml --task gsm8k \
        --t2l-dir trained_t2l/gemma_2b_t2l --seed 42 --max-examples -1

Design:
- Variants: baseline | generated | trained | openai, each with optional prompt
  prefix/suffix and temperature/shots overrides.
- Few-shot policies:
  - fixed: embed the same K demos into the prompt string across variants.
  - bootstrap: use DSPy BootstrapFewShot (same as scripts/evaluate.py).
- Runs variants sequentially, reusing a single TaskToLoRA/base model for local
  modes to keep VRAM usage low.
"""

from __future__ import annotations

import argparse
import inspect
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple

import dspy
import mlflow
import numpy as np
import pandas as pd
import torch
import yaml

# Ensure top-level project is importable
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dspy.teleprompt import BootstrapFewShot

# Reuse helpers from the single-run evaluator to avoid duplication
from scripts.evaluate import (  # type: ignore
    METRIC_FNS,
    TASK_CONFIGS,
    HFLocalLM,
    OpenAIChatLM,
    extract_gsm8k_answer,
    load_examples,
)
from src.dspy_task_to_lora import TaskToLoRA

# Match evaluate.py logging behavior
mlflow.dspy.autolog()
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("compare")


# Default task descriptions for Task-to-LoRA
TASK_DESCS: Dict[str, str] = {
    "gsm8k": "This task challenges your problem-solving abilities through mathematical reasoning. You must carefully read each scenario and systematically work through the data to compute the final outcome.",
    "medical": "Answer multiple-choice medical questions accurately.",
}


def set_seed(seed: int | None):
    if seed is None:
        return
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_fixed_fewshot_prefix(train_examples: List[dspy.Example], shots: int, *, seed: int | None = None) -> str:
    if shots <= 0:
        return ""
    # Deterministic subset (optionally shuffle by seed)
    idxs = list(range(min(shots, len(train_examples))))
    lines: List[str] = ["Here are some examples:"]
    for i in idxs:
        ex = train_examples[i]
        lines.append(f"Q{i+1}: {ex.prompt}\nA{i+1}: {ex.response}\n")
    lines.append("Now answer the following question:")
    return "\n".join(lines).strip() + "\n\n"


def compose_prompt(base: str, prefix: str = "", suffix: str = "", fewshot: str = "") -> str:
    parts = []
    if prefix:
        parts.append(prefix)
    if fewshot:
        parts.append(fewshot)
    parts.append(base)
    if suffix:
        parts.append(suffix)
    return "\n".join([p for p in parts if p])


def run_variant(
    *,
    variant: Dict[str, Any],
    task: str,
    task_desc: str,
    module: TaskToLoRA | None,
    train_examples: List[dspy.Example],
    dev_examples: List[dspy.Example],
    metric_fn,
    out_dir: Path,
    default_shots: int,
    fewshot_policy: str,
    max_tokens: int = 512,
):
    name: str = variant.get("name", variant.get("mode", "unnamed"))
    mode: str = variant["mode"]
    temperature: float = float(variant.get("temperature", 0.0))
    shots: int = int(variant.get("shots", default_shots))
    adapter_task_desc: str | None = variant.get("adapter_task_desc")
    prompt_cfg: Dict[str, Any] = variant.get("prompt", {}) or {}
    prefix: str = prompt_cfg.get("prefix", "")
    suffix: str = prompt_cfg.get("suffix", "")

    # Instantiate LM for this variant
    if mode == "openai":
        lm = OpenAIChatLM(model_id="gpt-4o-mini", max_tokens=max_tokens, temperature=temperature)
    elif mode == "baseline":
        if module is None:
            raise RuntimeError("Baseline requires a local TaskToLoRA module")
        try:
            module.base_model.set_adapter("default")
        except Exception:
            pass
        lm = HFLocalLM(
            module.base_model,
            module.tokenizer,
            max_tokens=max_tokens,
            temperature=temperature,
        )
    elif mode in {"generated", "trained"}:
        if module is None:
            raise RuntimeError(f"{mode} requires a local TaskToLoRA module")
        # Choose task description for adapter
        desc = adapter_task_desc or task_desc
        if mode == "generated":
            adapter_bundle = module(desc)
        else:
            artifact_path = Path(f"artifacts/{task}_trained.lora.pt")
            if not artifact_path.exists():
                raise FileNotFoundError(f"Trained adapter not found at {artifact_path}")
            adapter_bundle = torch.load(artifact_path)
        peft_model = module.apply_bundle(adapter_bundle, name=f"{mode}_{task}")
        lm = HFLocalLM(peft_model, module.tokenizer, max_tokens=max_tokens, temperature=temperature)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    dspy.settings.configure(lm=lm)

    signature = TASK_CONFIGS[task]["signature"]
    program = dspy.Predict(signature)

    compiled_program = program
    fewshot_prefix = ""

    # Optional tuner: GEPA prompt optimization
    tuner_cfg: Dict[str, Any] | None = variant.get("tuner")
    if tuner_cfg and str(tuner_cfg.get("type", "")).lower() == "gepa":
        compiled_program, tuned_artifacts = _maybe_run_gepa(
            program=program,
            tuner_cfg=tuner_cfg,
            metric_fn=metric_fn,
            train_examples=train_examples,
        )
        # Persist any tuned artifacts for inspection
        if tuned_artifacts:
            tuned_txt = tuned_artifacts.get("text")
            if tuned_txt:
                out_dir.mkdir(parents=True, exist_ok=True)
                prompt_path = out_dir / f"{task}_{name}_tuned_prompt.txt"
                try:
                    prompt_path.write_text(str(tuned_txt))
                    mlflow.log_artifact(str(prompt_path), artifact_path="results")
                except Exception:
                    pass

    # Few-shot handling if no tuner or in addition to tuner
    if shots > 0 and compiled_program is program:
        if fewshot_policy == "bootstrap":
            optimizer = BootstrapFewShot(metric=metric_fn, max_bootstrapped_demos=shots)
            compiled_program = optimizer.compile(program, trainset=train_examples[: shots * 2])
        else:  # fixed
            fewshot_prefix = build_fixed_fewshot_prefix(train_examples, shots)

    class _PredWrapper:
        def __init__(self, response: str | None):
            self.response = response

    rows: List[Dict[str, Any]] = []
    num_correct = 0

    # Create an MLflow run for the variant
    with mlflow.start_run(run_name=f"{task}_{name}_{shots}shots", nested=True):
        mlflow.log_params(
            {
                "task": task,
                "variant": name,
                "mode": mode,
                "shots": shots,
                "temperature": temperature,
                "fewshot_policy": fewshot_policy,
            }
        )

        for idx, example in enumerate(dev_examples):
            err = ""
            pred_text: str | None = None
            try:
                prompt_text = compose_prompt(example.prompt, prefix=prefix, suffix=suffix, fewshot=fewshot_prefix)
                pred = compiled_program(prompt=prompt_text)
                pred_text = getattr(pred, "response", None)
            except Exception as e:  # defensive: record and continue
                err = str(e)
                pred = _PredWrapper(None)

            if task == "gsm8k":
                pred_parsed = extract_gsm8k_answer(pred_text)
                gold_parsed = extract_gsm8k_answer(example.response)
            else:
                pred_parsed = (pred_text or "").strip()
                gold_parsed = example.response.strip()

            try:
                metric_val = float(metric_fn(example, _PredWrapper(pred_text)))
            except Exception:
                metric_val = 0.0

            num_correct += int(metric_val > 0.0)
            rows.append(
                {
                    "index": idx,
                    "prompt": example.prompt,
                    "gold": example.response,
                    "prediction": pred_text,
                    "gold_parsed": gold_parsed,
                    "pred_parsed": pred_parsed,
                    "correct": metric_val,
                    "error": err,
                }
            )

        score = num_correct / max(1, len(dev_examples))
        results = {
            "task": task,
            "variant": name,
            "mode": mode,
            "shots": shots,
            "score": score,
        }

        out_dir.mkdir(parents=True, exist_ok=True)
        summary_csv = out_dir / f"{task}_{name}_{shots}shots.csv"
        details_csv = out_dir / f"{task}_{name}_{shots}shots_details.csv"
        pd.DataFrame([results]).to_csv(summary_csv, index=False)
        pd.DataFrame(rows).to_csv(details_csv, index=False)

        mlflow.log_metric("average_metric", score)
        mlflow.log_artifact(str(summary_csv), artifact_path="results")
        mlflow.log_artifact(str(details_csv), artifact_path="results")

    print(f"[{name}] acc={score:.3f}; details -> {details_csv}")
    return details_csv, score


def aggregate(outputs: Dict[str, Path], task: str, spec_name: str, shots: int, out_dir: Path):
    variant_names = list(outputs.keys())
    if not variant_names:
        return

    # Create a wide table by merging on index; carry prompt/gold from the first variant
    base_name = variant_names[0]
    df_base = pd.read_csv(outputs[base_name])
    base_cols = [c for c in ["index", "prompt", "gold", "gold_parsed"] if c in df_base.columns]
    wide = df_base[base_cols].copy()

    for name in variant_names:
        df = pd.read_csv(outputs[name])
        keep_cols = ["index", "prediction", "pred_parsed", "correct"]
        keep_cols = [c for c in keep_cols if c in df.columns]
        df = df[keep_cols].rename(
            columns={
                "prediction": f"pred_{name}",
                "pred_parsed": f"pred_parsed_{name}",
                "correct": f"correct_{name}",
            }
        )
        wide = wide.merge(df, on="index", how="left")

    # Compute deltas vs the first variant
    base_col = f"correct_{base_name}"
    for name in variant_names[1:]:
        col = f"correct_{name}"
        if base_col in wide.columns and col in wide.columns:
            wide[f"delta_{name}_vs_{base_name}"] = wide[col] - wide[base_col]

    out_path = out_dir / f"{task}_{spec_name}_compare_{shots}shots.csv"
    wide.to_csv(out_path, index=False)

    # Log aggregated comparison to MLflow
    mlflow.log_artifact(str(out_path), artifact_path="results")

    # Console summary and analytics (enhanced visual formatting)
    total = len(wide)
    bar = "═" * 70
    sep = "-" * 70
    print("\n" + bar)
    print(" COMPARISON SUMMARY ".center(70, "═"))
    print(bar)
    print(f"{'Variant':<18} | {'Accuracy':>9} | {'Correct':>9} | {'Total':>5}")
    print(sep)
    for name in variant_names:
        ccol = f"correct_{name}"
        if ccol in wide.columns:
            acc = float(wide[ccol].mean()) if total else 0.0
            corr = int(wide[ccol].sum())
            print(f"{name:<18} | {acc:>9.2%} | {corr:>9}/{total:<5}")

    # Pairwise analysis for first two variants if present
    if len(variant_names) >= 2:
        m1, m2 = variant_names[0], variant_names[1]
        c1, c2 = f"correct_{m1}", f"correct_{m2}"
        print("\n" + bar)
        print(f" PAIRWISE ANALYSIS: {m1} vs {m2} ".center(70, "═"))
        print(bar)
        if c1 in wide.columns and c2 in wide.columns and total > 0:
            acc1 = float(wide[c1].mean())
            acc2 = float(wide[c2].mean())
            agree = int((wide[c1] == wide[c2]).sum())
            disagree = total - agree
            b_correct_l_wrong = int(((wide[c1] == 1.0) & (wide[c2] == 0.0)).sum())
            l_correct_b_wrong = int(((wide[c1] == 0.0) & (wide[c2] == 1.0)).sum())

            print(f"{m1:<18} | acc={acc1:>6.2%}    correct={int(wide[c1].sum())}/{total}")
            print(f"{m2:<18} | acc={acc2:>6.2%}    correct={int(wide[c2].sum())}/{total}")
            print(sep)
            print(f"Δ accuracy: {acc2 - acc1:+.2%}")
            print(f"Agree: {agree} ({agree/total:.1%}) | Disagree: {disagree} ({disagree/total:.1%})")
            print(f"  • {m1} correct, {m2} wrong: {b_correct_l_wrong}")
            print(f"  • {m2} correct, {m1} wrong: {l_correct_b_wrong}")

            # Show a few disagreements and save top disagreements
            mask_dis = wide[c1] != wide[c2]
            dis = wide[mask_dis].copy()
            if not dis.empty:
                print("\n" + bar)
                print(" EXAMPLE DISAGREEMENTS (first 3) ".center(70, "═"))
                print(bar)
                # Prefer parsed predictions if available
                p1p = f"pred_parsed_{m1}" if f"pred_parsed_{m1}" in dis.columns else f"pred_{m1}"
                p2p = f"pred_parsed_{m2}" if f"pred_parsed_{m2}" in dis.columns else f"pred_{m2}"
                gp = "gold_parsed" if "gold_parsed" in dis.columns else "gold"
                for _, row in dis.head(3).iterrows():
                    prompt_snip = row.get("prompt", "") or ""
                    if isinstance(prompt_snip, str) and len(prompt_snip) > 120:
                        prompt_snip = prompt_snip[:120] + "..."
                    print(f"\n• Example {int(row['index'])}")
                    print(f"  Prompt: {prompt_snip}")
                    print(f"  Gold:   {row.get(gp)}")
                    v1 = row.get(p1p)
                    v2 = row.get(p2p)
                    c1v = row.get(c1)
                    c2v = row.get(c2)
                    c1mark = "✓" if c1v == 1.0 else "✗"
                    c2mark = "✓" if c2v == 1.0 else "✗"
                    print(f"  {m1}: {v1} {c1mark}")
                    print(f"  {m2}: {v2} {c2mark}")

                # Save top disagreements to CSV for inspection
                top_n = min(10, len(dis))
                p1_save = p1p
                p2_save = p2p
                cols = ["index", "prompt", gp, p1_save, p2_save, c1, c2]
                save_df = (
                    dis[cols]
                    .rename(
                        columns={
                            gp: "gold_display",
                            p1_save: f"pred_{m1}",
                            p2_save: f"pred_{m2}",
                            c1: f"correct_{m1}",
                            c2: f"correct_{m2}",
                        }
                    )
                    .head(top_n)
                )
                top_path = out_dir / f"{task}_{spec_name}_compare_{shots}shots_top_disagreements.csv"
                save_df.to_csv(top_path, index=False)
                mlflow.log_artifact(str(top_path), artifact_path="results")
                print("\nTop disagreements ->", top_path)

    print("\nAggregated comparison ->", out_path)


def _maybe_run_gepa(
    *,
    program,
    tuner_cfg: Dict[str, Any],
    metric_fn,
    train_examples: List[dspy.Example],
) -> Tuple[Any, Dict[str, Any] | None]:
    """Attempt to run DSPy GEPA tuner, returning compiled_program and tuned artifacts.

    Falls back to returning the original program if GEPA is unavailable.
    """
    # Import GEPA from likely locations
    GEPA_cls = None
    err_msgs: List[str] = []
    for path in (
        "dspy.optimize",
        "dspy.teleprompt",
        "dspy.experimental.optimize",
    ):
        try:
            mod = __import__(path, fromlist=["GEPA"])  # type: ignore
            GEPA_cls = getattr(mod, "GEPA", None)
            if GEPA_cls is not None:
                break
        except Exception as e:
            err_msgs.append(f"{path}: {e}")

    if GEPA_cls is None:
        warnings.warn(
            "GEPA optimizer not found in DSPy. Skipping tuner. "
            "Try upgrading dspy and see tutorials: https://dspy.ai/tutorials/gepa_ai_program/",
            RuntimeWarning,
        )
        return program, None

    # Prepare training subset
    train_size = int(tuner_cfg.get("train_size", min(256, len(train_examples))))
    trainset = train_examples[:train_size]

    # Map tuner config to GEPA __init__ kwargs using signature filtering
    desired_kwargs = {
        "metric": metric_fn,
        "metric_fn": metric_fn,
        "iters": int(tuner_cfg.get("iters", 10)),
        "n_iter": int(tuner_cfg.get("iters", 10)),
        "candidates": int(tuner_cfg.get("candidates", 8)),
        "n_candidates": int(tuner_cfg.get("candidates", 8)),
        "scope": tuner_cfg.get("scope", "prefix"),
        "judge": tuner_cfg.get("judge", "self"),
        "seed": tuner_cfg.get("seed", None),
    }

    try:
        init_sig = inspect.signature(GEPA_cls.__init__)
        allowed = set(init_sig.parameters.keys())
        kwargs = {k: v for k, v in desired_kwargs.items() if k in allowed}
    except Exception:
        kwargs = {k: v for k, v in desired_kwargs.items() if v is not None}

    try:
        optimizer = GEPA_cls(**kwargs)
    except Exception as e:
        warnings.warn(f"Failed to initialize GEPA with kwargs {kwargs}: {e}", RuntimeWarning)
        return program, None

    # Call compile with flexible arg names
    try:
        compile_sig = inspect.signature(optimizer.compile)
        params = set(compile_sig.parameters.keys())
        c_kwargs = {}
        if "trainset" in params:
            c_kwargs["trainset"] = trainset
        elif "train_data" in params:
            c_kwargs["train_data"] = trainset
        elif "train_examples" in params:
            c_kwargs["train_examples"] = trainset
        compiled_program = optimizer.compile(program, **c_kwargs)
    except Exception as e:
        warnings.warn(f"GEPA compile failed: {e}", RuntimeWarning)
        return program, None

    # Try to extract tuned text artifacts if present
    tuned: Dict[str, Any] = {}
    for attr in ("best_prompt", "prompt", "instructions", "text"):
        val = getattr(optimizer, attr, None)
        if isinstance(val, str) and val.strip():
            tuned["text"] = val
            break
    return compiled_program, (tuned or None)


def main():
    parser = argparse.ArgumentParser(description="Flexible comparison orchestrator")
    parser.add_argument("--spec", required=True, help="Path to YAML spec with variants")
    parser.add_argument("--task", choices=["gsm8k", "medical"], required=True)
    parser.add_argument("--t2l-dir", default=str(ROOT / "trained_t2l" / "gemma_2b_t2l"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-examples", type=int, default=-1)
    parser.add_argument("--fewshot-policy", choices=["fixed", "bootstrap"], default="fixed")
    args = parser.parse_args()

    set_seed(args.seed)

    with open(args.spec, "r") as f:
        spec = yaml.safe_load(f)

    spec_name: str = spec.get("name", Path(args.spec).stem)
    variants: List[Dict[str, Any]] = spec.get("variants", [])
    default_shots: int = int(spec.get("shots", 0))

    task = args.task
    config = TASK_CONFIGS[task]
    metric_fn = METRIC_FNS[config["metric"]]
    task_desc = TASK_DESCS[task]

    data_dir = ROOT / "data" / task
    train_path = data_dir / f"{config['dataset_prefix']}_train.jsonl"
    dev_path = data_dir / f"{config['dataset_prefix']}_dev.jsonl"
    train_examples = load_examples(train_path)
    dev_examples = load_examples(dev_path)
    if args.max_examples and args.max_examples > 0:
        dev_examples = dev_examples[: args.max_examples]

    # Build a single TaskToLoRA for local modes
    module: TaskToLoRA | None = TaskToLoRA(args.t2l_dir, return_model=False, seed=args.seed)

    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)

    # Parent MLflow run for all variants (nice grouping in UI)
    with mlflow.start_run(run_name=f"compare_{task}_{spec_name}"):
        mlflow.log_params(
            {
                "compare_spec": spec_name,
                "task": task,
                "seed": args.seed,
                "fewshot_policy": args.fewshot_policy,
                "max_examples": args.max_examples,
            }
        )

        outputs: Dict[str, Path] = {}
        for v in variants:
            name = v.get("name", v.get("mode", "unnamed"))
            details_path, _ = run_variant(
                variant=v,
                task=task,
                task_desc=task_desc,
                module=module,
                train_examples=train_examples,
                dev_examples=dev_examples,
                metric_fn=metric_fn,
                out_dir=results_dir,
                default_shots=default_shots,
                fewshot_policy=args.fewshot_policy,
            )
            outputs[name] = details_path

        # Aggregate results while still within MLflow context
        aggregate(
            outputs,
            task=task,
            spec_name=spec_name,
            shots=default_shots,
            out_dir=results_dir,
        )


if __name__ == "__main__":
    main()
