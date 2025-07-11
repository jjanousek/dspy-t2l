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
from rouge_score import rouge_scorer
from sacrebleu import sentence_bleu
from transformers import AutoModelForCausalLM, AutoTokenizer

# Assuming TaskToLoRA is in src/ or importable
from src.dspy_task_to_lora import TaskToLoRA  # Adjust path as per your repo


# Define DSPy signatures per task (simple ChainOfThought for reasoning/generation)
class BasicQA(dspy.Signature):
    """Answer the question based on the prompt."""

    prompt: str = dspy.InputField()
    response: str = dspy.OutputField()


class CodeGeneration(dspy.Signature):
    """Generate executable Python code to solve the problem."""

    prompt: str = dspy.InputField()
    response: str = dspy.OutputField(desc="Python code snippet")


class MedicalQA(dspy.Signature):
    """Select the correct answer letter for the medical question."""

    prompt: str = dspy.InputField()
    response: str = dspy.OutputField(desc="Single letter: A, B, C, or D")


# class SalesEmail(dspy.Signature):
#     """Generate a professional sales email."""
#     prompt: str = dspy.InputField()
#     response: str = dspy.OutputField(desc="Full email body")

# Map tasks to signatures and metrics
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
    # "sales": {
    #     "signature": SalesEmail,
    #     "metric": "rouge",  # ROUGE-L for text generation
    #     "dataset_prefix": "synthetic_sales",
    # },
}


# Metric functions
def exact_match_metric(example, pred, trace=None):
    return example.response.strip().upper() == pred.response.strip().upper()


def bleu_metric(example, pred, trace=None):
    ref = example.response
    hyp = pred.response
    return sentence_bleu(hyp, [ref]).score / 100  # Normalize to 0-1


def rouge_metric(example, pred, trace=None):
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = scorer.score(example.response, pred.response)
    return scores["rougeL"].fmeasure  # F1 score


METRIC_FNS = {
    "accuracy": exact_match_metric,
    "bleu": bleu_metric,
    "rouge": rouge_metric,
}


# Load examples from JSONL
def load_examples(file_path: Path) -> list[dspy.Example]:
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


# Main evaluation function
def run_evaluation(args):
    task = args.task
    mode = args.mode
    shots = args.shots
    config = TASK_CONFIGS[task]
    metric_fn = METRIC_FNS[config["metric"]]

    # Load datasets
    data_dir = Path(f"data/{task}")
    train_path = data_dir / f"{config['dataset_prefix']}_train.jsonl"
    dev_path = data_dir / f"{config['dataset_prefix']}_dev.jsonl"
    # test_path = data_dir / f"{config['dataset_prefix']}_test.jsonl"

    train_examples = load_examples(train_path)
    dev_examples = load_examples(dev_path)
    # test_examples = load_examples(
    #     test_path
    # )  # Use dev for quick eval; swap to test for final

    # Task description for hyper-network
    task_descs = {
        "code": "Generate Python code to solve math reasoning problems like those in GSM8K.",
        "medical": "Answer multiple-choice medical questions accurately.",
        # "sales": "Write persuasive sales emails for products to various customers.",
    }
    task_desc = task_descs[task]

    # --------------------------------------------------------------
    # 1. Obtain the adapter bundle *first* so we can figure out which
    #    base model to load (Gemma-2B in the default checkpoint).
    # --------------------------------------------------------------
    hypermod_dir = args.t2l_dir  # directory holding hypermod.pt & config
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

    # Ensure we have a PEFT config object
    if isinstance(adapter_bundle["config"], PeftConfig):
        peft_config = adapter_bundle["config"]
    else:
        peft_config = PeftConfig.from_dict(adapter_bundle["config"])

    # --------------------------------------------------------------
    # 2.  Load the correct base model (typically Gemma-2B) specified
    #     inside the adapter's PEFT config.
    # --------------------------------------------------------------
    base_model_id = (
        peft_config.base_model_name_or_path
        if hasattr(peft_config, "base_model_name_or_path")
        else peft_config["base_model_name_or_path"]
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        load_in_4bit=True,
        device_map="auto",
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )

    # Apply adapter to the freshly-loaded base model
    peft_model = get_peft_model(base_model, peft_config)
    peft_model.load_state_dict(adapter_bundle["state_dict"], strict=False)

    # Setup DSPy LM with the PEFT model
    lm = dspy.HFModel(
        model=peft_model, tokenizer=tokenizer, max_tokens=512
    )  # Adjust params as needed
    dspy.settings.configure(lm=lm)

    # Define program
    program = dspy.ChainOfThought(config["signature"])  # Or Predict for simpler

    # Few-shot optimization if shots > 0
    if shots > 0:
        optimizer = BootstrapFewShot(metric=metric_fn, max_bootstrapped_demos=shots)
        compiled_program = optimizer.compile(
            program, trainset=train_examples[: shots * 2]
        )  # Use some for validation internally
    else:
        compiled_program = program

    # Evaluate on dev/test set
    evaluator = Evaluate(
        devset=dev_examples,  # Or test_examples for final
        metric=metric_fn,
        num_threads=4,  # Parallelism
        display_progress=True,
    )
    score = evaluator(compiled_program)

    # Collect results
    results = {
        "task": task,
        "mode": mode,
        "shots": shots,
        "score": score,
        "metric": config["metric"],
    }

    # Output
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    csv_path = results_dir / f"{task}_{mode}_{shots}shots.csv"
    df = pd.DataFrame([results])
    df.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")
    print(df.to_markdown(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate LoRA adapters with DSPy.")
    parser.add_argument(
        "--mode", choices=["generated", "trained"], required=True, help="Adapter mode"
    )
    # parser.add_argument("--task", choices=["code", "medical", "sales"], required=True, help="Task to evaluate")
    parser.add_argument(
        "--task", choices=["code", "medical"], required=True, help="Task to evaluate"
    )
    parser.add_argument(
        "--shots",
        type=int,
        default=0,
        help="Number of few-shot examples (0 for zero-shot)",
    )
    # Default to the Gemma-2B Task-to-LoRA checkpoint that ships with the repo
    default_ckpt = Path(__file__).resolve().parents[1] / "trained_t2l" / "gemma_2b_t2l"
    parser.add_argument(
        "--t2l_dir",
        default=str(default_ckpt),
        help="Path to a trained_t2l/<checkpoint> folder containing hypermod.pt and adapter_config.json "
        f"(default: {default_ckpt})",
    )
    args = parser.parse_args()

    run_evaluation(args)
