#!/usr/bin/env python
# ruff: noqa: E402
"""Debug why generation is stopping early with RSLoRA."""

import sys
from pathlib import Path

import torch

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import re

from peft import PeftConfig, set_peft_model_state_dict

from src.dspy_task_to_lora import TaskToLoRA


def debug_generation_stopping():
    """Debug why RSLoRA outputs are truncated."""

    print("=" * 80)
    print("DEBUGGING GENERATION STOPPING")
    print("=" * 80)

    # Initialize without the fix
    def apply_bundle_with_rslora(self, bundle, name="default"):
        """Apply bundle WITHOUT disabling RSLoRA."""
        peft_config = bundle["config"]
        if not isinstance(peft_config, PeftConfig):
            peft_config = PeftConfig.from_dict(peft_config)

        if name not in self.peft_model.peft_config:
            self.peft_model.add_adapter(name, peft_config)

        state_dict = bundle["state_dict"]
        normalized_sd = {}
        model_dtype = next(self.base_model.parameters()).dtype
        for k, v in state_dict.items():
            new_k = re.sub(r"(\.lora_[AB])\.[^.]+(\.weight)$", r"\1\2", k)
            if v.dtype != model_dtype:
                v = v.to(dtype=model_dtype)
            normalized_sd[new_k] = v

        set_peft_model_state_dict(self.peft_model, normalized_sd, adapter_name=name)
        self.peft_model.set_adapter(name)
        return self.peft_model

    # Save original and replace
    original_apply = TaskToLoRA.apply_bundle
    TaskToLoRA.apply_bundle = apply_bundle_with_rslora

    try:
        t2l_dir = ROOT / "trained_t2l" / "gemma_2b_t2l"
        module = TaskToLoRA(t2l_dir, return_model=False, seed=42)
        tokenizer = module.tokenizer

        # Check special tokens
        print("\n1. Special tokens in tokenizer:")
        print(f"   eos_token: {tokenizer.eos_token} (id: {tokenizer.eos_token_id})")
        print(f"   pad_token: {tokenizer.pad_token} (id: {tokenizer.pad_token_id})")
        if hasattr(tokenizer, "additional_special_tokens"):
            print(f"   additional_special_tokens: {tokenizer.additional_special_tokens}")

        # Check for end_of_turn token
        end_of_turn_id = None
        if "<end_of_turn>" in tokenizer.get_vocab():
            end_of_turn_id = tokenizer.convert_tokens_to_ids("<end_of_turn>")
            print(f"   <end_of_turn> token id: {end_of_turn_id}")

        # Generate and apply RSLoRA
        task_desc = "This task challenges your problem-solving abilities through mathematical reasoning. You must carefully read each scenario and systematically work through the data to compute the final outcome."
        bundle = module.forward(task_desc)
        peft_model = module.apply_bundle(bundle, name="gsm8k_rslora")

        # Test prompt
        test_prompt = "Please answer the following question: Lorraine and Colleen are trading stickers for buttons. Each large sticker is worth a large button or three small buttons. A small sticker is worth one small button. A large button is worth three small stickers. Lorraine starts with 30 small stickers and 40 large stickers. She trades 90% of her small stickers for large buttons. She trades 50% of her large stickers for large buttons and trades the rest of them for small buttons. How many buttons does she have by the end?"

        # Format with chat template
        if hasattr(tokenizer, "chat_template") and tokenizer.chat_template:
            messages = [{"role": "user", "content": test_prompt}]
            formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            formatted = test_prompt

        print("\n2. Formatted prompt:")
        print(f"   {repr(formatted[:200])}")

        inputs = tokenizer(formatted, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(module.device) for k, v in inputs.items()}

        print("\n3. Testing different generation configurations:")

        # Test 1: Default generation
        print("\n   a) Default generation (max_new_tokens=100):")
        with torch.no_grad():
            outputs = peft_model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                return_dict_in_generate=True,
                output_scores=True,
            )

        generated_ids = outputs.sequences[0][len(inputs["input_ids"][0]) :]
        response = tokenizer.decode(generated_ids, skip_special_tokens=False)
        response_clean = tokenizer.decode(generated_ids, skip_special_tokens=True)

        print(f"      Raw output (with special tokens): {repr(response)}")
        print(f"      Clean output: {response_clean}")
        print(f"      Generated token IDs: {generated_ids.tolist()[:20]}")

        # Check if end_of_turn token was generated
        if end_of_turn_id and end_of_turn_id in generated_ids:
            position = (generated_ids == end_of_turn_id).nonzero()[0].item()
            print(f"      ⚠️  Found <end_of_turn> at position {position}")

        # Test 2: Without early stopping
        print("\n   b) With eos_token_id=None (no early stopping):")
        with torch.no_grad():
            outputs_no_stop = peft_model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=False,
                eos_token_id=None,  # Disable early stopping
                pad_token_id=tokenizer.pad_token_id,
            )

        response_no_stop = tokenizer.decode(
            outputs_no_stop[0][len(inputs["input_ids"][0]) :], skip_special_tokens=True
        )
        print(f"      Output: {response_no_stop[:200]}")

        # Test 3: With explicit stopping criteria
        print("\n   c) With min_new_tokens=50:")
        with torch.no_grad():
            outputs_min = peft_model.generate(
                **inputs,
                max_new_tokens=100,
                min_new_tokens=50,  # Force at least 50 tokens
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        response_min = tokenizer.decode(outputs_min[0][len(inputs["input_ids"][0]) :], skip_special_tokens=True)
        print(f"      Output: {response_min[:200]}")

        # Test 4: Check what tokens are being generated
        print("\n   d) Token-by-token generation (first 10 tokens):")
        with torch.no_grad():
            current_ids = inputs["input_ids"]
            for i in range(10):
                outputs = peft_model(input_ids=current_ids)
                next_token_logits = outputs.logits[0, -1, :]
                next_token_id = torch.argmax(next_token_logits).unsqueeze(0).unsqueeze(0)
                next_token = tokenizer.decode(next_token_id[0], skip_special_tokens=False)

                print(f"      Token {i+1}: {repr(next_token)} (id: {next_token_id.item()})")

                # Check if it's a stopping token
                if next_token_id.item() == tokenizer.eos_token_id:
                    print("         ⚠️  This is the EOS token!")
                if end_of_turn_id and next_token_id.item() == end_of_turn_id:
                    print("         ⚠️  This is the <end_of_turn> token!")

                current_ids = torch.cat([current_ids, next_token_id], dim=1)

        # Compare with base model
        print("\n4. Comparing with BASE model:")
        module.peft_model.set_adapter("default")

        with torch.no_grad():
            base_outputs = module.peft_model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        base_response = tokenizer.decode(base_outputs[0][len(inputs["input_ids"][0]) :], skip_special_tokens=False)
        print(f"   Base model raw output: {repr(base_response[:100])}")

    finally:
        # Restore original method
        TaskToLoRA.apply_bundle = original_apply

    print("\n" + "=" * 80)
    print("KEY FINDINGS:")
    print("The RSLoRA model is likely generating an early stopping token")
    print("(either EOS or <end_of_turn>) right after the number.")
    print("=" * 80)


if __name__ == "__main__":
    debug_generation_stopping()
