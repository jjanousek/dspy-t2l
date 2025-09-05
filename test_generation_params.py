#!/usr/bin/env python3
"""Test if generation parameters affect the truncation issue."""

from pathlib import Path

import torch

from scripts.evaluate import HFLocalLM
from src.dspy_task_to_lora import TaskToLoRA

# Initialize
t2l_dir = Path("trained_t2l/gemma_2b_t2l")
module = TaskToLoRA(t2l_dir, return_model=False, seed=42)

task_desc = "This task challenges your problem-solving abilities through mathematical reasoning. You must carefully read each scenario and systematically work through the data to compute the final outcome."

test_prompt = "Question: John has 5 apples. He gives 2 to Mary. How many apples does John have left?\n\nAnswer:"

print("=" * 70)
print("TESTING GENERATION PARAMETERS")
print("=" * 70)

# Apply generated adapter
adapter_bundle = module(task_desc)
peft_model = module.apply_bundle(adapter_bundle, name="generated_gsm8k")

# Test different max_tokens values
max_token_values = [50, 100, 256, 512, 1024]

for max_tok in max_token_values:
    print(f"\n--- max_tokens = {max_tok} ---")
    lm = HFLocalLM(peft_model, module.tokenizer, max_tokens=max_tok, temperature=0.0)
    result = lm(prompt=test_prompt)
    print(f"Response: {result[0]}")
    print(f"Length: {len(result[0])} chars")

# Test with different generation parameters
print("\n--- Testing with do_sample=False explicitly ---")
# Check the actual generation kwargs in HFLocalLM
print("\nChecking HFLocalLM generation method...")

# Direct generation test
print("\n--- Direct model.generate() test ---")
inputs = module.tokenizer(test_prompt, return_tensors="pt").to(peft_model.device)
with torch.inference_mode():
    # Match Sakana's settings exactly
    outputs = peft_model.generate(
        **inputs,
        max_new_tokens=1024,
        temperature=0.0,
        do_sample=False,
        pad_token_id=module.tokenizer.eos_token_id,
    )
completion = module.tokenizer.decode(outputs[0][len(inputs["input_ids"][0]) :], skip_special_tokens=True)
print(f"Direct generation response: {completion}")
print(f"Length: {len(completion)} chars")

print("\n" + "=" * 70)
