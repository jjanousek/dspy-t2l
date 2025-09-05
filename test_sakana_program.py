#!/usr/bin/env python3
"""Test the SakanaChatProgram to see if it's causing the issue."""

import re
from pathlib import Path

import dspy

from scripts.evaluate import HFLocalLM
from src.dspy_task_to_lora import TaskToLoRA
from src.sakana_chat_program import SakanaChatProgram

# Initialize the TaskToLoRA module
t2l_dir = Path("trained_t2l/gemma_2b_t2l")
module = TaskToLoRA(t2l_dir, return_model=False, seed=42)

task_desc = "This task challenges your problem-solving abilities through mathematical reasoning. You must carefully read each scenario and systematically work through the data to compute the final outcome."

# Sakana's fixed ICL examples
SAKANA_GSM8K_ICL = (
    "Here are some examples of the tasks you will be asked to solve.\n\n"
    "## Example 1\n"
    "Question: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?\n\n"
    "Answer: Natalia sold 48/2 = <<48/2=24>>24 clips in May.\n"
    "Natalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.\n"
    "#### 72\n\n"
    "## Example 2\n"
    "Question: Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?\n\n"
    "Answer: Weng earns 12/60 = $<<12/60=0.2>>0.2 per minute.\n"
    "Working 50 minutes, she earned 0.2 x 50 = $<<0.2*50=10>>10.\n"
    "#### 10\n\n"
    "## Example 3\n"
    "Question: Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet?\n\n"
    "Answer: In the beginning, Betty has only 100 / 2 = $<<100/2=50>>50.\n"
    "Betty's grandparents gave her 15 * 2 = $<<15*2=30>>30.\n"
    "This means, Betty needs 100 - 50 - 30 - 15 = $<<100-50-30-15=5>>5 more.\n"
    "#### 5\n\n"
)

test_question = "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?"

print("=" * 70)
print("TESTING SAKANA CHAT PROGRAM")
print("=" * 70)

# Create the SakanaChatProgram
program = SakanaChatProgram(
    icl_block=SAKANA_GSM8K_ICL,
    include_icl=True,
    prefix="",
    suffix="",
)

# Build messages to inspect them
messages = program.build_messages(test_question)
print("\n1. Messages built by SakanaChatProgram:")
for msg in messages:
    print(f"   Role: {msg['role']}")
    print(f"   Content preview: {msg['content'][:100]}...")

# Test with baseline adapter
print("\n2. BASELINE with SakanaChatProgram:")
module.base_model.set_adapter("default")
lm_baseline = HFLocalLM(module.base_model, module.tokenizer, max_tokens=200, temperature=0.0)
dspy.settings.configure(lm=lm_baseline)

pred_baseline = program(prompt=test_question)
print(f"   Response: {pred_baseline.response[:200]}...")

# Test with generated adapter
print("\n3. GENERATED with SakanaChatProgram:")
adapter_bundle = module(task_desc)
peft_model = module.apply_bundle(adapter_bundle, name="generated_gsm8k")
lm_generated = HFLocalLM(peft_model, module.tokenizer, max_tokens=200, temperature=0.0)
dspy.settings.configure(lm=lm_generated)

pred_generated = program(prompt=test_question)
print(f"   Response: {pred_generated.response[:200]}...")

# Now test WITHOUT SakanaChatProgram (direct LM call with messages)
print("\n4. DIRECT LM CALL (without SakanaChatProgram):")
# Apply chat template directly
formatted_prompt = module.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
print(f"   Formatted prompt length: {len(formatted_prompt)} chars")

# Test with generated adapter
result_direct = lm_generated(messages=messages)
print(f"   Direct response: {result_direct[0][:200]}...")

# Extract answers


def extract_answer(text):
    if "####" in text:
        after = text.split("####")[-1].strip()
        matches = re.findall(r"(-?\d+(?:,\d+)*(?:\.\d+)?)", after)
        if matches:
            try:
                return float(matches[0].replace(",", ""))
            except Exception:
                pass
    matches = re.findall(r"(-?\d+(?:,\d+)*(?:\.\d+)?)", text)
    if matches:
        try:
            return float(matches[-1].replace(",", ""))
        except Exception:
            pass
    return None


print("\n5. EXTRACTED ANSWERS:")
print(f"   Baseline: {extract_answer(pred_baseline.response)}")
print(f"   Generated: {extract_answer(pred_generated.response)}")
print(f"   Direct: {extract_answer(result_direct[0])}")
print("   Correct: 18")

print("\n" + "=" * 70)
