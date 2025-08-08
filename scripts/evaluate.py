# scripts/evaluate.py
"""
Evaluation script for Task-Adaptive LoRAs.

Usage:
    python evaluate.py --mode generated --task code --shots 5

This script evaluates LoRA adapters (generated or trained) on specified tasks using DSPy.
It loads datasets from data/{task}/, runs inference with few-shot if specified,
and computes task-appropriate metrics. Outputs results as CSV and Markdown table.
"""

import argparse
import json
import random
import re
from pathlib import Path

import dspy
import mlflow
import numpy as np
import openai
import pandas as pd
import torch
from dspy.teleprompt import BootstrapFewShot

from src.dspy_task_to_lora import TaskToLoRA

mlflow.dspy.autolog()

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("evaluate")

try:
    # Use Sakana helper that attaches the correct chat_template for the model
    from hyper_llm_modulator.utils.model_loading import get_tokenizer  # type: ignore
except ImportError:
    # hyper_llm_modulator may not be installed in some environments; fall back to HF
    get_tokenizer = None  # type: ignore


class HFLocalLM(dspy.LM):
    """Custom DSPy LM provider for locally-loaded Hugging Face models."""

    def __init__(
        self,
        model,
        tokenizer,
        max_tokens: int = 512,
        temperature: float = 0.0,
        top_p: float = 0.9,
    ):
        super().__init__(model="local_hf")
        self.hf_model = model
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        self.kwargs = {
            "temperature": temperature,
            "top_p": top_p,
            "do_sample": temperature > 0,
        }
        self.history = []

    def _generate(self, prompt: str, max_tokens: int, **kwargs) -> str:
        # Apply chat template if available; otherwise fall back to raw prompt
        if getattr(self.tokenizer, "chat_template", None):
            messages = [{"role": "user", "content": prompt}]
            prompt = self.tokenizer.apply_chat_template(messages, tokenize=False)

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.hf_model.device)
        with torch.inference_mode():
            outputs = self.hf_model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                pad_token_id=self.tokenizer.eos_token_id,
                **kwargs,
            )
        completion = self.tokenizer.decode(
            outputs[0][len(inputs["input_ids"][0]) :], skip_special_tokens=True
        )
        return completion.strip()

    def basic_request(self, prompt: str, **kwargs):
        merged_kwargs = {**self.kwargs, **kwargs}
        response_text = self._generate(prompt, self.max_tokens, **merged_kwargs)
        return [{"role": "assistant", "content": response_text}]

    def __call__(
        self,
        prompt: str | None = None,
        *,
        messages: list | None = None,
        only_completed: bool = True,
        return_sorted: bool = False,
        **kwargs,
    ):
        if messages is not None and prompt is None:
            # Flatten chat messages into a single prompt string if provided as a list
            prompt = "\n".join([m.get("content", "") for m in messages])
        elif prompt is None:
            raise ValueError("Either `prompt` or `messages` must be provided.")

        response = self.basic_request(prompt, **kwargs)
        self.history.append({"prompt": prompt, "response": response, "kwargs": kwargs})
        if only_completed:
            return [r["content"] for r in response]
        return response


class OpenAIChatLM(dspy.LM):
    # DSPy LM wrapper around OpenAI ChatCompletion API, shaped to mirror HFLocalLM.
    def __init__(
        self,
        model_id: str = "o3",
        max_tokens: int = 512,
        temperature: float = 0.0,
        top_p: float = 0.9,
    ):
        super().__init__(model=f"openai_{model_id}")
        self.model_id = model_id
        self.max_tokens = max_tokens
        self.kwargs = {
            "temperature": temperature,
            "top_p": top_p,
        }
        self.history = []

    def basic_request(
        self, prompt: str | None = None, *, messages: list | None = None, **kwargs
    ):
        merged_kwargs = {**self.kwargs, **kwargs}
        if messages is None:
            # Fallback: wrap prompt in a single user message
            if prompt is None:
                raise ValueError("Either `prompt` or `messages` must be provided.")
            messages = [{"role": "user", "content": prompt}]
        response = openai.chat.completions.create(
            model=self.model_id,
            messages=messages,
            max_tokens=self.max_tokens,
            **merged_kwargs,
        )
        text = response.choices[0].message.content.strip()
        return [{"role": "assistant", "content": text}]

    def __call__(
        self,
        prompt: str | None = None,
        *,
        messages: list | None = None,
        only_completed: bool = True,
        return_sorted: bool = False,
        **kwargs,
    ):
        if messages is None and prompt is None:
            raise ValueError("Either `prompt` or `messages` must be provided.")

        response = self.basic_request(prompt, messages=messages, **kwargs)
        self.history.append({"prompt": prompt, "response": response, "kwargs": kwargs})
        if only_completed:
            return [r["content"] for r in response]
        return response


class MathReasoning(dspy.Signature):
    """Solve the math problem by showing your work step-by-step."""

    prompt: str = dspy.InputField()
    response: str = dspy.OutputField(
        desc="A detailed, step-by-step answer to the math problem, ending with '#### <answer>'."
    )


class MedicalQA(dspy.Signature):
    """Select the correct answer letter for the medical question."""

    prompt: str = dspy.InputField()
    response: str = dspy.OutputField(desc="Single letter: A, B, C, or D")


TASK_CONFIGS = {
    "gsm8k": {
        "signature": MathReasoning,
        "metric": "gsm8k_accuracy",
        "dataset_prefix": "gsm8k",
    },
    "medical": {
        "signature": MedicalQA,
        "metric": "accuracy",  # Exact match on letter
        "dataset_prefix": "medmcqa",
    },
}


def exact_match_metric(example, pred, trace=None):
    """Binary exact-match metric tolerant to missing model outputs."""
    # Handle edge cases where the model fails to return a prediction
    if getattr(pred, "response", None) in (None, ""):
        return 0.0  # count as incorrect
    return float(example.response.strip().upper() == pred.response.strip().upper())


def extract_gsm8k_answer(text: str | None) -> float | None:
    """Extract the last number from text as a float."""
    if text is None:
        return None

    text = text.replace("\n", "")

    matches = re.findall(r"\d*\.?\d+", text)
    if not matches:
        return None
    try:
        return float(matches[-1].replace(",", ""))
    except ValueError:
        return None


def gsm8k_metric(example, pred, trace=None):
    """Binary accuracy using numeric parsing."""
    pred_val = extract_gsm8k_answer(getattr(pred, "response", None))
    gold_val = extract_gsm8k_answer(example.response)

    if pred_val is None or gold_val is None:
        return 0.0
    return float(pred_val == gold_val)


METRIC_FNS = {
    "accuracy": exact_match_metric,
    "gsm8k_accuracy": gsm8k_metric,
}


def load_examples(file_path: Path) -> list[dspy.Example]:
    """Load a JSONL file into a list of DSPy examples."""
    examples = []
    with open(file_path, "r") as f:
        for line in f:
            ex = json.loads(line)
            examples.append(
                dspy.Example(prompt=ex["prompt"], response=ex["response"]).with_inputs(
                    "prompt"
                )
            )
    return examples


def run_evaluation(args):
    # Set seed for reproducibility
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    task = args.task
    mode = args.mode
    shots = args.shots

    config = TASK_CONFIGS[task]
    metric_fn = METRIC_FNS[config["metric"]]

    # load data
    data_dir = Path(f"data/{task}")
    train_path = data_dir / f"{config['dataset_prefix']}_train.jsonl"
    dev_path = data_dir / f"{config['dataset_prefix']}_dev.jsonl"
    # test_path = (
    #     data_dir / f"{config['dataset_prefix']}_test.jsonl"
    # )  # not used by default

    train_examples = load_examples(train_path)
    dev_examples = load_examples(dev_path)
    # Optionally limit the number of evaluation examples for faster debugging
    if getattr(args, "max_examples", -1) and args.max_examples > 0:
        dev_examples = dev_examples[: args.max_examples]

    # task-to-LoRA inference
    task_descs = {
        "gsm8k": "Solve grade-school math problems by showing your work step-by-step, ending with the final answer in the format '#### <number>'.",
        "medical": "Answer multiple-choice medical questions accurately.",
    }
    task_desc = task_descs[task]

    hypermod_dir = args.t2l_dir

    # Only load the Hyper-LoRA stack when we actually need it. This avoids
    # pulling the (large) base model weights into memory for the OpenAI path.
    module = None  # type: ignore[assignment]
    if mode != "openai":
        module = TaskToLoRA(hypermod_dir, return_model=False, seed=args.seed)

    if mode == "openai":
        lm = OpenAIChatLM(
            model_id="gpt-4o-mini",
            max_tokens=512,
            temperature=args.temperature,
        )
    elif mode == "baseline":
        # Directly use the underlying base model from the TaskToLoRA checkpoint
        lm = HFLocalLM(
            module.base_model,
            module.tokenizer,
            max_tokens=512,
            temperature=args.temperature,
        )
    else:
        if mode == "generated":
            adapter_bundle = module.forward(task_desc)
        elif mode == "trained":
            artifact_path = Path(f"artifacts/{task}_trained.lora.pt")
            if not artifact_path.exists():
                raise FileNotFoundError(f"Trained adapter not found at {artifact_path}")
            adapter_bundle = torch.load(artifact_path)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        # apply adapter to existing model
        # NOTE: We reuse the PEFT model from the TaskToLoRA instance to avoid loading
        # a duplicate base model
        peft_model = module.apply_bundle(adapter_bundle, name=f"{mode}_{task}")
        peft_model.load_state_dict(adapter_bundle["state_dict"], strict=False)

        # wrap with custom DSPy LM
        lm = HFLocalLM(
            peft_model, module.tokenizer, max_tokens=512, temperature=args.temperature
        )

    dspy.settings.configure(lm=lm)

    # build program
    # use a simple predictor instead of ChainOfThought to avoid JSON parsing issues
    program = dspy.Predict(config["signature"])

    if shots > 0:
        optimizer = BootstrapFewShot(metric=metric_fn, max_bootstrapped_demos=shots)
        compiled_program = optimizer.compile(
            program, trainset=train_examples[: shots * 2]
        )
    else:
        compiled_program = program

    # Evaluate per-example to compute metrics and log detailed outputs
    class _PredWrapper:
        def __init__(self, response: str | None):
            self.response = response

    per_example_rows: list[dict] = []
    num_correct = 0

    # Start a top-level MLflow run so we can attach artifacts alongside traces
    with mlflow.start_run(run_name=f"{task}_{mode}_{shots}shots"):
        mlflow.log_params(
            {
                "task": task,
                "mode": mode,
                "shots": shots,
                "temperature": args.temperature,
                "max_examples": args.max_examples,
                "metric": config["metric"],
            }
        )

        for idx, example in enumerate(dev_examples):
            error_msg = ""
            pred_text: str | None = None
            try:
                pred = compiled_program(prompt=example.prompt)
                pred_text = getattr(pred, "response", None)
            except Exception as e:  # defensive: record error and continue
                error_msg = str(e)
                pred = _PredWrapper(None)

            # Compute parsed values and metric
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
            per_example_rows.append(
                {
                    "index": idx,
                    "prompt": example.prompt,
                    "gold": example.response,
                    "prediction": pred_text,
                    "gold_parsed": gold_parsed,
                    "pred_parsed": pred_parsed,
                    "correct": metric_val,
                    "error": error_msg,
                }
            )

        score = num_correct / max(1, len(dev_examples))

        # results
        results = {
            "task": task,
            "mode": mode,
            "shots": shots,
            "score": score,
            "metric": config["metric"],
        }

        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)

        summary_csv_path = results_dir / f"{task}_{mode}_{shots}shots.csv"
        pd.DataFrame([results]).to_csv(summary_csv_path, index=False)

        details_df = pd.DataFrame(per_example_rows)
        # Put incorrect rows first for quick visibility
        details_df_sorted = details_df.sort_values(["correct", "index"])  # 0s then 1s
        details_csv_path = results_dir / f"{task}_{mode}_{shots}shots_details.csv"
        details_df_sorted.to_csv(details_csv_path, index=False)

        # Log to MLflow for convenient browsing
        mlflow.log_metric("average_metric", score)
        mlflow.log_artifact(str(summary_csv_path), artifact_path="results")
        mlflow.log_artifact(str(details_csv_path), artifact_path="results")

    print(f"Results saved to {summary_csv_path}")
    print(pd.DataFrame([results]).to_markdown(index=False))
    print(f"Per-example details saved to {details_csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate LoRA adapters with DSPy.")
    parser.add_argument(
        "--mode",
        choices=["generated", "trained", "baseline", "openai"],
        required=True,
        help="Adapter mode ('baseline' runs without any LoRA)",
    )
    parser.add_argument(
        "--task", choices=["gsm8k", "medical"], required=True, help="Task to evaluate"
    )
    parser.add_argument(
        "--shots",
        type=int,
        default=0,
        help="Number of few-shot examples (0 for zero-shot)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Temperature for sampling. Set to 0.0 for deterministic output.",
    )

    # Limit the number of evaluation examples to speed up quick tests. Use -1 for all.
    parser.add_argument(
        "--max_examples",
        type=int,
        default=3,
        help="Maximum dev examples to evaluate (-1 for all examples)",
    )

    default_ckpt = Path(__file__).resolve().parents[1] / "trained_t2l" / "gemma_2b_t2l"
    parser.add_argument(
        "--t2l_dir",
        default=str(default_ckpt),
        help="Path to a trained_t2l/<checkpoint> folder containing hypermod.pt and adapter_config.json",
    )

    run_evaluation(parser.parse_args())
