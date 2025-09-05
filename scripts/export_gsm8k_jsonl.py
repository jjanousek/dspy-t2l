#!/usr/bin/env python
"""
Export the full GSM8K dataset (train and test) to JSONL files used by this repo.

Outputs:
  - data/gsm8k/gsm8k_train.jsonl  (7,473 examples)
  - data/gsm8k/gsm8k_test.jsonl   (1,319 examples)

Each line is a JSON object with keys:
  - prompt:   "Please answer the following question: {question}"
  - response:  the original GSM8K "answer" field (contains reasoning + #### final)

Usage:
  uv run python scripts/export_gsm8k_jsonl.py

Notes:
  - Requires internet/HF cache access for datasets.load_dataset("gsm8k", "main").
  - Run `huggingface-cli login` if needed.
"""

from __future__ import annotations

import json
from pathlib import Path

from datasets import load_dataset


def _write_split(ds, out_path: Path) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out_path.open("w") as f:
        for ex in ds:
            q = ex["question"].strip()
            a = ex["answer"].strip()
            prompt = f"Please answer the following question: {q}"
            obj = {"prompt": prompt, "response": a}
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
            n += 1
    return n


def main():
    ds_train = load_dataset("gsm8k", "main", split="train")
    ds_test = load_dataset("gsm8k", "main", split="test")

    out_dir = Path("data/gsm8k")
    n_train = _write_split(ds_train, out_dir / "gsm8k_train.jsonl")
    n_test = _write_split(ds_test, out_dir / "gsm8k_test.jsonl")

    print(f"Wrote train: {n_train} examples -> {out_dir/'gsm8k_train.jsonl'}")
    print(f"Wrote test:  {n_test} examples -> {out_dir/'gsm8k_test.jsonl'}")


if __name__ == "__main__":
    main()
