#!/usr/bin/env python
"""Quick test to verify Sakana parity setup."""

import json
from pathlib import Path

# Test answer extraction
from scripts.evaluate import extract_gsm8k_answer

# Test cases from Sakana documentation
test_cases = [
    # With #### delimiter
    ("Janet has 16 eggs. After using 7, she has 16 - 7 = 9 eggs. She sells 9 * $2 = $18.\n#### 18", 18.0),
    # Without delimiter, last number
    (
        "1. **Eggs left after eating:** 16 - 3 = 13 eggs\n2. **Eggs for muffins:** 13 - 4 = 9 eggs\n3. **Daily earnings:** 9 * $2 = $18\n\nJanet makes **$18** every day.",
        18.0,
    ),
    # With commas
    ("The total is 1,234 dollars", 1234.0),
    # Negative numbers
    ("The profit is -50", -50.0),
    # Decimals
    ("The answer is 12.5", 12.5),
]

print("Testing answer extraction (Sakana parity):")
print("-" * 50)
for text, expected in test_cases:
    result = extract_gsm8k_answer(text)
    status = "✓" if result == expected else f"✗ (got {result})"
    print(f"{status} Expected {expected}: {text[:50]}...")

# Load and display a sample GSM8K example
print("\n" + "=" * 50)
print("Sample GSM8K training example format:")
print("-" * 50)
train_path = Path("data/gsm8k/gsm8k_train.jsonl")
with open(train_path) as f:
    example = json.loads(f.readline())
    print(f"Prompt: {example['prompt'][:100]}...")
    print(f"Response: {example['response'][:150]}...")

print("\n" + "=" * 50)
print("To run with Sakana parity, use:")
print(
    "uv run python scripts/compare.py --spec configs/compare_gsm8k_sakana.yaml --task gsm8k --t2l-dir trained_t2l/gemma_2b_t2l --fewshot-policy fixed --max-examples -1"
)
print("\nThis will:")
print("1. Use 3-shot fixed ICL (matching Sakana’s examples style)")
print("2. Apply the exact task description for LoRA generation")
print("3. Use improved answer extraction (#### delimiter support)")
print("4. Run on full test set (-1 = all examples)")
