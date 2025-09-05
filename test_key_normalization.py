#!/usr/bin/env python3
"""Test the key normalization issue in apply_bundle."""

import re
from pathlib import Path

from src.dspy_task_to_lora import TaskToLoRA

# Initialize
t2l_dir = Path("trained_t2l/gemma_2b_t2l")
module = TaskToLoRA(t2l_dir, return_model=False, seed=42)

task_desc = "This task challenges your problem-solving abilities through mathematical reasoning."

print("=" * 70)
print("KEY NORMALIZATION ANALYSIS")
print("=" * 70)

# Generate adapter
adapter_bundle = module(task_desc)

print("\n1. Original state dict keys (first 5):")
for i, key in enumerate(list(adapter_bundle["state_dict"].keys())[:5]):
    print(f"   {key}")

print("\n2. After regex normalization (current implementation):")
for i, key in enumerate(list(adapter_bundle["state_dict"].keys())[:5]):
    new_k = re.sub(r"(\.lora_[AB])\.[^.]+(\.weight)$", r"\1\2", key)
    print(f"   {key} -> {new_k}")

print("\n3. Expected PEFT format for a named adapter:")
print("   For adapter named 'test_adapter', PEFT expects:")
print("   base_model.model.model.layers.0.self_attn.q_proj.lora_A.test_adapter.weight")

print("\n4. Checking what PEFT actually expects:")
# Add a test adapter manually to see the expected format
test_config = adapter_bundle["config"]
module.peft_model.add_adapter("test_adapter", test_config)

# Get the state dict of the model and look for test_adapter keys
all_keys = module.peft_model.state_dict().keys()
test_adapter_keys = [k for k in all_keys if "test_adapter" in k and "lora" in k][:3]
print("   Actual PEFT keys for 'test_adapter':")
for key in test_adapter_keys:
    print(f"   {key}")

print("\n5. Testing different normalization strategies:")

# Strategy 1: Keep adapter name (no normalization)
print("\n   Strategy 1 - No normalization:")
sample_key = "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight"
print(f"   Input: {sample_key}")
print(f"   Output: {sample_key} (unchanged)")

# Strategy 2: Current regex (strips adapter name)
print("\n   Strategy 2 - Current regex (strips everything between lora_X and weight):")
sample_key = "base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight"
new_k = re.sub(r"(\.lora_[AB])\.[^.]+(\.weight)$", r"\1\2", sample_key)
print(f"   Input: {sample_key}")
print(f"   Output: {new_k}")

# Strategy 3: Replace 'default' with target adapter name
print("\n   Strategy 3 - Replace adapter name:")
sample_key = "base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight"
target_name = "generated_gsm8k"
new_k = sample_key.replace(".default.", f".{target_name}.")
print(f"   Input: {sample_key}")
print(f"   Output: {new_k}")

print("\n" + "=" * 70)
