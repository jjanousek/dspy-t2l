import json
from pathlib import Path

from datasets import load_dataset

# Load full dataset
ds = load_dataset("medmcqa")

# Subsample: 200 from train (for our 'train' split), 100 from validation (split into dev/test)
train_subset = ds["train"].shuffle(seed=42).select(range(200))
val_subset = ds["validation"].shuffle(seed=42).select(range(100))

# Process into prompt-response pairs
train_examples = []
for ex in train_subset:
    options = f"A. {ex['opa']} B. {ex['opb']} C. {ex['opc']} D. {ex['opd']}"
    prompt = f"Answer this medical question with the correct letter: {ex['question']}\nOptions: {options}\nAnswer:"
    response = chr(ord("A") + ex["cop"])  # 0=A, 1=B, etc.
    train_examples.append({"prompt": prompt, "response": response})

val_examples = []
for ex in val_subset:
    options = f"A. {ex['opa']} B. {ex['opb']} C. {ex['opc']} D. {ex['opd']}"
    prompt = f"Answer this medical question with the correct letter: {ex['question']}\nOptions: {options}\nAnswer:"
    response = chr(ord("A") + ex["cop"])
    val_examples.append({"prompt": prompt, "response": response})

# Split val_subset into dev (first 50) and test (next 50)
dev_examples = val_examples[:50]
test_examples = val_examples[50:]

# Save
Path("data/medical/").mkdir(parents=True, exist_ok=True)
for split, data in [
    ("train", train_examples),
    ("dev", dev_examples),
    ("test", test_examples),
]:
    with open(f"data/medical/medmcqa_{split}.jsonl", "w") as f:
        for ex in data:
            f.write(json.dumps(ex) + "\n")
