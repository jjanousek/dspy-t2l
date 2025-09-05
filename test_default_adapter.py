#!/usr/bin/env python3
"""Check what the default adapter actually contains."""

from pathlib import Path

import torch

from scripts.evaluate import HFLocalLM
from src.dspy_task_to_lora import TaskToLoRA

# Initialize the TaskToLoRA module
t2l_dir = Path("trained_t2l/gemma_2b_t2l")
module = TaskToLoRA(t2l_dir, return_model=False, seed=42)

print("=" * 70)
print("DEFAULT ADAPTER ANALYSIS")
print("=" * 70)

# Check the default adapter
print("\n1. Model has these adapters:", list(module.peft_model.peft_config.keys()))
print("2. Active adapter:", module.peft_model.active_adapters)

# Get the default adapter weights
print("\n3. Default adapter weight analysis:")
module.peft_model.set_adapter("default")

# Sample some weights from the default adapter
for name, param in module.peft_model.named_parameters():
    if "lora" in name and "default" in name:
        print(f"\n   Parameter: {name}")
        print(f"   Shape: {param.shape}")
        print(f"   Dtype: {param.dtype}")
        print(f"   Mean: {param.mean().item():.6f}")
        print(f"   Std: {param.std().item():.6f}")
        print(f"   Min/Max: {param.min().item():.6f} / {param.max().item():.6f}")

        # Check if it's actually zeros (identity adapter)
        is_zero = torch.allclose(param, torch.zeros_like(param), atol=1e-6)
        print(f"   Is zero tensor: {is_zero}")
        break

# Now disable all adapters to get true base model
print("\n4. Attempting to disable all adapters...")
try:
    module.peft_model.disable_adapter()
    print("   Successfully disabled adapters")
    print("   Active adapters:", module.peft_model.active_adapters)
except Exception as e:
    print(f"   Error disabling adapters: {e}")

# Test inference with and without adapter
test_prompt = "Question: John has 5 apples. He gives 2 to Mary. How many apples does John have left?\n\nAnswer:"

print("\n5. Testing generation WITH default adapter:")
module.peft_model.set_adapter("default")
lm_with = HFLocalLM(module.peft_model, module.tokenizer, max_tokens=100, temperature=0.0)
result_with = lm_with(prompt=test_prompt)
print(f"   Response: {result_with[0][:200]}")

print("\n6. Testing generation WITHOUT any adapter:")
module.peft_model.disable_adapter()
lm_without = HFLocalLM(module.peft_model, module.tokenizer, max_tokens=100, temperature=0.0)
result_without = lm_without(prompt=test_prompt)
print(f"   Response: {result_without[0][:200]}")

print("\n" + "=" * 70)
