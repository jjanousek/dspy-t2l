import json
from pathlib import Path

from datasets import load_dataset

# Load full dataset
ds = load_dataset("gsm8k", "main")

# Subsample: 200 from train (for our 'train' split), 100 from test (split into dev/test)
train_subset = ds["train"].shuffle(seed=42).select(range(200))
test_subset = ds["test"].shuffle(seed=42).select(range(100))

# Process into prompt-response pairs for math reasoning
train_examples = []
for ex in train_subset:
    prompt = f"Please answer the following question: {ex['question']}"
    response = f"Let's think step by step. {ex['answer']}"
    train_examples.append({"prompt": prompt, "response": response})

test_examples = []
for ex in test_subset:
    prompt = f"Please answer the following question: {ex['question']}"
    response = f"Let's think step by step. {ex['answer']}"
    test_examples.append({"prompt": prompt, "response": response})

# Split test_subset into dev (first 50) and test (next 50)
dev_examples = test_examples[:50]
test_examples = test_examples[50:]

# Save
Path("data/gsm8k/").mkdir(parents=True, exist_ok=True)
for split, data in [
    ("train", train_examples),
    ("dev", dev_examples),
    ("test", test_examples),
]:
    with open(f"data/gsm8k/gsm8k_{split}.jsonl", "w") as f:
        for ex in data:
            f.write(json.dumps(ex) + "\n")
