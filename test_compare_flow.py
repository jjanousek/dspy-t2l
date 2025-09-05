#!/usr/bin/env python3
"""Test that mimics the exact flow in compare.py to identify the bug."""

import re
from pathlib import Path

from scripts.evaluate import HFLocalLM
from src.dspy_task_to_lora import TaskToLoRA

# Initialize the TaskToLoRA module (shared across variants like in compare.py)
t2l_dir = Path("trained_t2l/gemma_2b_t2l")
module = TaskToLoRA(t2l_dir, return_model=False, seed=42)

task_desc = "This task challenges your problem-solving abilities through mathematical reasoning. You must carefully read each scenario and systematically work through the data to compute the final outcome."

test_prompt = """Here are some examples of the tasks you will be asked to solve.

## Example 1
Question: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?

Answer: Natalia sold 48/2 = <<48/2=24>>24 clips in May.
Natalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.
#### 72

Please answer the following question: Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?"""

print("=" * 70)
print("MIMICKING COMPARE.PY FLOW")
print("=" * 70)

# STEP 1: Run baseline (like compare.py does)
print("\n1. BASELINE MODE (mimics lines 169-180 of compare.py):")
print("   Setting adapter to 'default'...")
try:
    module.base_model.set_adapter("default")
except Exception as e:
    print(f"   Exception (caught): {e}")
    pass

print(f"   Active adapter: {module.base_model.active_adapters}")
lm_baseline = HFLocalLM(module.base_model, module.tokenizer, max_tokens=100, temperature=0.0)

result_baseline = lm_baseline(prompt=test_prompt)
print(f"   Response preview: {result_baseline[0][:150]}...")

# STEP 2: Run generated mode (like compare.py does)
print("\n2. GENERATED MODE (mimics lines 181-194 of compare.py):")
print("   Generating adapter...")
adapter_bundle = module(task_desc)

print("   Applying bundle with name 'generated_gsm8k'...")
peft_model = module.apply_bundle(adapter_bundle, name="generated_gsm8k")

# Check if it's the same object
print(f"   peft_model is module.base_model: {peft_model is module.base_model}")
print(f"   peft_model is module.peft_model: {peft_model is module.peft_model}")
print(f"   Active adapter after apply: {peft_model.active_adapters}")

lm_generated = HFLocalLM(peft_model, module.tokenizer, max_tokens=100, temperature=0.0)

result_generated = lm_generated(prompt=test_prompt)
print(f"   Response preview: {result_generated[0][:150]}...")

# STEP 3: Try running baseline AGAIN (simulating sequential runs)
print("\n3. BASELINE MODE AGAIN (checking for state pollution):")
print("   Setting adapter back to 'default'...")
module.base_model.set_adapter("default")
print(f"   Active adapter: {module.base_model.active_adapters}")

lm_baseline2 = HFLocalLM(module.base_model, module.tokenizer, max_tokens=100, temperature=0.0)
result_baseline2 = lm_baseline2(prompt=test_prompt)
print(f"   Response preview: {result_baseline2[0][:150]}...")

# Compare results
print("\n4. COMPARISON:")
print(f"   Baseline 1st run == Baseline 2nd run: {result_baseline[0] == result_baseline2[0]}")
print(f"   Baseline == Generated: {result_baseline[0] == result_generated[0]}")

# Extract numerical answers


def extract_answer(text):
    matches = re.findall(r"(-?\d+(?:,\d+)*(?:\.\d+)?)", text)
    if matches:
        try:
            return float(matches[-1].replace(",", ""))
        except Exception:
            pass
    return None


ans_baseline = extract_answer(result_baseline[0])
ans_generated = extract_answer(result_generated[0])
print(f"\n   Baseline answer: {ans_baseline}")
print(f"   Generated answer: {ans_generated}")
print("   Correct answer: 18")

print("\n" + "=" * 70)
