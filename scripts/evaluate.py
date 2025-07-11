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
from pathlib import Path

import dspy
import pandas as pd
import torch
from dspy.evaluate import Evaluate
from dspy.teleprompt import BootstrapFewShot
from peft import PeftConfig, get_peft_model
from sacrebleu import sentence_bleu
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    # Use Sakana helper that attaches the correct chat_template for the model
    from hyper_llm_modulator.utils.model_loading import get_tokenizer  # type: ignore
except ImportError:
    # hyper_llm_modulator may not be installed in some environments; fall back to HF
    get_tokenizer = None  # type: ignore

# Assuming TaskToLoRA is in src/ or importable
from src.dspy_task_to_lora import TaskToLoRA  # Adjust path as per your repo


# -----------------------------------------------------------------------------
# Custom LM class for local HF models (dspy.HFModel deprecated in latest DSPy)
# -----------------------------------------------------------------------------
class HFLocalLM(dspy.LM):
    """Custom DSPy LM provider for locally-loaded Hugging Face models."""

    def __init__(
        self,
        model,
        tokenizer,
        max_tokens: int = 512,
        temperature: float = 0.7,
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

    # Internal generate helper -------------------------------------------------
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

    # DSPy LM interface --------------------------------------------------------
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
        # DSPy expects LM.__call__(prompt=None, messages=None, **kwargs)
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


# -----------------------------------------------------------------------------
# DSPy signatures per task
# -----------------------------------------------------------------------------
class CodeGeneration(dspy.Signature):
    """Generate executable Python code to solve the problem."""

    prompt: str = dspy.InputField()
    response: str = dspy.OutputField(desc="Python code snippet")


class MedicalQA(dspy.Signature):
    """Select the correct answer letter for the medical question."""

    prompt: str = dspy.InputField()
    response: str = dspy.OutputField(desc="Single letter: A, B, C, or D")


# Map tasks to signatures and metrics -----------------------------------------
TASK_CONFIGS = {
    "code": {
        "signature": CodeGeneration,
        "metric": "bleu",  # BLEU for code similarity
        "dataset_prefix": "gsm8k_code",
    },
    "medical": {
        "signature": MedicalQA,
        "metric": "accuracy",  # Exact match on letter
        "dataset_prefix": "medmcqa",
    },
}

# Metric functions -------------------------------------------------------------


def exact_match_metric(example, pred, trace=None):
    """Binary exact-match metric tolerant to missing model outputs."""
    # Handle edge cases where the model fails to return a prediction
    if getattr(pred, "response", None) in (None, ""):
        return 0.0  # count as incorrect
    return float(example.response.strip().upper() == pred.response.strip().upper())


def bleu_metric(example, pred, trace=None):
    """Sentence-level BLEU score scaled to 0-1, robust to empty predictions."""
    # If prediction is missing we treat it as zero-quality output
    if getattr(pred, "response", None) in (None, ""):
        return 0.0

    # Normalize whitespace to avoid spurious BLEU penalties
    ref = " ".join(example.response.split()) if example.response else ""
    hyp = " ".join(pred.response.split())

    # If reference happens to be empty (should not occur), return 0
    if ref == "" or hyp == "":
        return 0.0

    return sentence_bleu(hyp, [ref]).score / 100  # Scale to 0-1 range


METRIC_FNS = {
    "accuracy": exact_match_metric,
    "bleu": bleu_metric,
}


# Utility ---------------------------------------------------------------------


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


# Main evaluation --------------------------------------------------------------


def run_evaluation(args):
    task = args.task
    mode = args.mode
    shots = args.shots

    config = TASK_CONFIGS[task]
    metric_fn = METRIC_FNS[config["metric"]]

    # ------------------------------ Load data ---------------------------------
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

    # ----------------------- Task-to-LoRA inference ---------------------------
    task_descs = {
        "code": "Generate Python code to solve math reasoning problems like those in GSM8K.",
        "medical": "Answer multiple-choice medical questions accurately.",
    }
    task_desc = task_descs[task]

    hypermod_dir = args.t2l_dir
    module = TaskToLoRA(hypermod_dir, return_model=False)

    if mode == "generated":
        adapter_bundle = module.forward(task_desc)
    elif mode == "trained":
        artifact_path = Path(f"artifacts/{task}_trained.lora.pt")
        if not artifact_path.exists():
            raise FileNotFoundError(f"Trained adapter not found at {artifact_path}")
        adapter_bundle = torch.load(artifact_path)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # ------------------------- Load base model --------------------------------
    if isinstance(adapter_bundle["config"], PeftConfig):
        peft_config = adapter_bundle["config"]
    else:
        peft_config = PeftConfig.from_dict(adapter_bundle["config"])

    base_model_id = peft_config.base_model_name_or_path

    # Load tokenizer with chat template support when available
    if get_tokenizer is not None:
        tokenizer = get_tokenizer(base_model_id)
    else:
        tokenizer = AutoTokenizer.from_pretrained(base_model_id)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        load_in_4bit=True,
        device_map="auto",
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )

    # ------------------------ Apply PEFT adapter ------------------------------
    peft_model = get_peft_model(base_model, peft_config)
    peft_model.load_state_dict(adapter_bundle["state_dict"], strict=False)

    # -------------------- Wrap with custom DSPy LM ----------------------------
    lm = HFLocalLM(peft_model, tokenizer, max_tokens=512)
    dspy.settings.configure(lm=lm)

    # --------------------------- Build program --------------------------------
    # Use a simple predictor instead of ChainOfThought to avoid JSON parsing issues
    program = dspy.Predict(config["signature"])

    if shots > 0:
        optimizer = BootstrapFewShot(metric=metric_fn, max_bootstrapped_demos=shots)
        compiled_program = optimizer.compile(
            program, trainset=train_examples[: shots * 2]
        )
    else:
        compiled_program = program

    # ----------------------------- Evaluate -----------------------------------
    evaluator = Evaluate(
        devset=dev_examples,
        metric=metric_fn,
        num_threads=4,
        display_progress=True,
    )
    score = evaluator(compiled_program)

    # ------------------------------ Results -----------------------------------
    results = {
        "task": task,
        "mode": mode,
        "shots": shots,
        "score": score,
        "metric": config["metric"],
    }

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    csv_path = results_dir / f"{task}_{mode}_{shots}shots.csv"
    pd.DataFrame([results]).to_csv(csv_path, index=False)

    print(f"Results saved to {csv_path}")
    print(pd.DataFrame([results]).to_markdown(index=False))


# -----------------------------------------------------------------------------
# Entry-point ------------------------------------------------------------------
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate LoRA adapters with DSPy.")
    parser.add_argument(
        "--mode", choices=["generated", "trained"], required=True, help="Adapter mode"
    )
    parser.add_argument(
        "--task", choices=["code", "medical"], required=True, help="Task to evaluate"
    )
    parser.add_argument(
        "--shots",
        type=int,
        default=0,
        help="Number of few-shot examples (0 for zero-shot)",
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
