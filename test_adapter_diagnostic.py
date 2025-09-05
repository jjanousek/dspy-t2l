#!/usr/bin/env python3
"""Diagnostic script to test LoRA adapter application and generation."""

from pathlib import Path

from scripts.evaluate import HFLocalLM
from src.dspy_task_to_lora import TaskToLoRA

# Initialize the TaskToLoRA module
t2l_dir = Path("trained_t2l/gemma_2b_t2l")
module = TaskToLoRA(t2l_dir, return_model=False, seed=42)

# Task description from Sakana
task_desc = "This task challenges your problem-solving abilities through mathematical reasoning. You must carefully read each scenario and systematically work through the data to compute the final outcome."

print("=" * 70)
print("DIAGNOSTIC: Testing LoRA Adapter Application")
print("=" * 70)

# Check base model state
print("\n1. Base Model Info:")
print(f"   Model type: {type(module.base_model).__name__}")
print(f"   Model ID: {module.base_model_id}")
print(f"   Has PEFT: {hasattr(module.base_model, 'peft_config')}")
if hasattr(module.base_model, "peft_config"):
    print(f"   PEFT configs: {list(module.base_model.peft_config.keys())}")
    print(f"   Active adapters: {module.base_model.active_adapters}")

# Generate the adapter
print("\n2. Generating adapter for task...")
adapter_bundle = module(task_desc)

print(f"   State dict keys (first 5): {list(adapter_bundle['state_dict'].keys())[:5]}")
print(f"   State dict size: {len(adapter_bundle['state_dict'])} tensors")

# Check a sample tensor
sample_key = list(adapter_bundle["state_dict"].keys())[0]
sample_tensor = adapter_bundle["state_dict"][sample_key]
print("\n3. Sample tensor analysis:")
print(f"   Key: {sample_key}")
print(f"   Shape: {sample_tensor.shape}")
print(f"   Dtype: {sample_tensor.dtype}")
print(f"   Device: {sample_tensor.device}")
print(f"   Mean: {sample_tensor.mean().item():.6f}")
print(f"   Std: {sample_tensor.std().item():.6f}")
print(f"   Min/Max: {sample_tensor.min().item():.6f} / {sample_tensor.max().item():.6f}")

# Apply the adapter
print("\n4. Applying adapter with name 'generated_gsm8k'...")
peft_model = module.apply_bundle(adapter_bundle, name="generated_gsm8k")

print(f"   PEFT configs after apply: {list(peft_model.peft_config.keys())}")
print(f"   Active adapters after apply: {peft_model.active_adapters}")

# Check if weights are actually loaded
print("\n5. Checking if adapter weights are properly loaded:")
for name, param in peft_model.named_parameters():
    if "lora" in name and "generated_gsm8k" in name:
        print(f"   Found LoRA param: {name}")
        print(f"      Shape: {param.shape}, Dtype: {param.dtype}")
        print(f"      Mean: {param.mean().item():.6f}, Std: {param.std().item():.6f}")
        break

# Test with baseline (switching adapters)
print("\n6. Testing adapter switching:")
if "default" in peft_model.peft_config:
    peft_model.set_adapter("default")
    print(f"   Switched to 'default': {peft_model.active_adapters}")

peft_model.set_adapter("generated_gsm8k")
print(f"   Switched back to 'generated_gsm8k': {peft_model.active_adapters}")

# Generate some text to verify it's working
print("\n7. Test generation with adapter:")

lm = HFLocalLM(peft_model, module.tokenizer, max_tokens=50, temperature=0.0)

test_prompt = "Question: What is 2 + 2?\n\nAnswer:"
result = lm(prompt=test_prompt)
print(f"   Prompt: {test_prompt}")
print(f"   Response: {result[0][:200]}")

print("\n" + "=" * 70)
print("DIAGNOSTIC COMPLETE")
print("=" * 70)
