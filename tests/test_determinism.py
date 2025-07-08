# ruff: noqa: E402
import sys
from pathlib import Path

import torch

root_dir = Path(__file__).resolve().parents[1]
src_dir = root_dir / "src"
if src_dir not in map(Path, map(Path, sys.path)):
    sys.path.insert(0, str(src_dir))

from dspy_task_to_lora import TaskToLoRA


def test_determinism(tmp_path):
    # Use the Gemma-2B Task-to-LoRA checkpoint that ships with the repo
    checkpoint_dir = root_dir / "trained_t2l" / "gemma_2b_t2l"
    mod = TaskToLoRA(str(checkpoint_dir), device="cpu")
    a = mod("Solve grade-school math")["state_dict"]
    b = mod("Solve grade-school math")["state_dict"]
    for k in a:
        assert torch.equal(a[k], b[k]), f"Mismatch in {k}"
