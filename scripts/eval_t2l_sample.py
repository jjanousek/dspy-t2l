# ruff: noqa: E402
import argparse
import sys
from pathlib import Path

import torch

# Make sure the top-level "src/" directory is on the module search path
root_dir = Path(__file__).resolve().parents[1]
src_dir = root_dir / "src"
if src_dir not in map(Path, map(Path, sys.path)):
    sys.path.insert(0, str(src_dir))

from peft import get_peft_model
from peft.utils import set_peft_model_state_dict  # NEW
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from dspy_task_to_lora import TaskToLoRA

TASKS = [
    "Solve grade-school math word problems",
    "Detect sentiment (positive/negative) in short reviews",
    "Translate from English to German",
]


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---------- 1.   Prepare components -----------------------------
    t2l = TaskToLoRA(args.t2l_dir, device=device)

    # Pick the right base model for this checkpoint
    model_id = t2l.peft_cfg.base_model_name_or_path  # e.g. Llama-3-8B

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token_id is None:  # avoid warnings at generation
        tokenizer.pad_token_id = tokenizer.eos_token_id

    max_memory = {0: "20GiB", "cpu": "50GiB"}  # adjust to your GPU/host
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        load_in_4bit=True,  # keep 4-bit quantisation
        device_map="auto",
        max_memory=max_memory,  # prevent unwanted offload
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )

    for task in TASKS:
        print(f"\n=== {task} ===")
        # ---------- 2.   Obtain adapter ------------------------------
        adapter = t2l(task)  # returns {"state_dict", "config"}
        model = get_peft_model(base_model, adapter["config"], adapter_name="generated")
        set_peft_model_state_dict(
            model, adapter["state_dict"], adapter_name="generated"
        )
        model.set_adapter("generated")
        text_gen = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
        )

        # ---------- 3.   Run a single demo prompt --------------------
        if "sentiment" in task.lower():
            q = "Review: 'The movie was really not bad at all! I couldn't stop laughing!'\nSentiment:"
        elif "translate" in task.lower():
            q = "Translate to German: 'The weather is nice today.'"
        else:
            q = "If a farm has 3 chickens and each lays 2 eggs, how many eggs in total?"

        out = text_gen(
            q,
            max_new_tokens=30,
            pad_token_id=tokenizer.eos_token_id,
        )[
            0
        ]["generated_text"]
        print(out.strip())


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    # Default to the Gemma-2B checkpoint, but allow overriding via CLI
    default_ckpt = Path(__file__).resolve().parents[1] / "trained_t2l" / "gemma_2b_t2l"
    p.add_argument(
        "--t2l_dir",
        default=str(default_ckpt),
        help="Path to a trained_t2l/<checkpoint> folder containing "
        "hypermod.pt and adapter_config.json "
        f"(default: {default_ckpt})",
    )
    main(p.parse_args())
