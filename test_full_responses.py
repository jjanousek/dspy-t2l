#!/usr/bin/env python3
"""Get full responses to understand why generated adapter performs poorly."""

from pathlib import Path

import dspy

from scripts.evaluate import HFLocalLM
from src.dspy_task_to_lora import TaskToLoRA
from src.sakana_chat_program import SakanaChatProgram

# Initialize
t2l_dir = Path("trained_t2l/gemma_2b_t2l")
module = TaskToLoRA(t2l_dir, return_model=False, seed=42)

task_desc = "This task challenges your problem-solving abilities through mathematical reasoning. You must carefully read each scenario and systematically work through the data to compute the final outcome."

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
print("FULL RESPONSE COMPARISON")
print("=" * 70)

program = SakanaChatProgram(
    icl_block=SAKANA_GSM8K_ICL,
    include_icl=True,
    prefix="",
    suffix="",
)

# Baseline
print("\n1. BASELINE FULL RESPONSE:")
print("-" * 50)
module.base_model.set_adapter("default")
lm_baseline = HFLocalLM(module.base_model, module.tokenizer, max_tokens=512, temperature=0.0)
dspy.settings.configure(lm=lm_baseline)
pred_baseline = program(prompt=test_question)
print(pred_baseline.response)
print("-" * 50)

# Generated
print("\n2. GENERATED ADAPTER FULL RESPONSE:")
print("-" * 50)
adapter_bundle = module(task_desc)
peft_model = module.apply_bundle(adapter_bundle, name="generated_gsm8k")
lm_generated = HFLocalLM(peft_model, module.tokenizer, max_tokens=512, temperature=0.0)
dspy.settings.configure(lm=lm_generated)
pred_generated = program(prompt=test_question)
print(pred_generated.response)
print("-" * 50)

print("\n3. CORRECT SOLUTION:")
print("-" * 50)
print(
    """Janet's ducks lay 16 eggs per day.
She eats 3 eggs for breakfast: 16 - 3 = 13 eggs remaining.
She uses 4 eggs for muffins: 13 - 4 = 9 eggs remaining.
She sells 9 eggs at $2 each: 9 × $2 = $18.
#### 18"""
)
print("-" * 50)

print("\n" + "=" * 70)
