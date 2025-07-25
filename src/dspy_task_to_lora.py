from __future__ import annotations

import copy
import threading
from pathlib import Path
from typing import Any, Dict

import dspy
import torch
from hyper_llm_modulator.hyper_modulator import load_hypermod_checkpoint
from hyper_llm_modulator.utils import embed_texts, get_layers
from peft import (
    PeftConfig,
    get_peft_config,
    get_peft_model,
    set_peft_model_state_dict,
)


class TaskToLoRA(dspy.Module):
    """Generate LoRA adapters from plain-English task descriptions.

    Parameters
    ----------
    hypermod_dir : str | Path
        Path to a directory that contains:
        - ``hypermod.pt`` - serialized hyper-network weights
        - ``adapter_config.json`` - PEFT LoRA config JSON
        - auxiliary tokenizer + embedding files produced by SakanaAI script
    device : str | torch.device, default "cuda"
        Where to place the hyper-network and intermediate tensors.
    cache_size : int, default 32
        Maximum number of distinct task descriptions whose adapters are
        kept in memory at once. New entries evict the oldest (LRU).
    return_model : bool, default False
        If True, :py:meth:`forward` returns a fully materialized
        ``peft.LoraModel`` instance. Otherwise it returns a lightweight
        ``{"state_dict", "config"}`` bundle which callers can attach to
        any compatible base model.
    """

    def __init__(
        self,
        hypermod_dir: str | Path,
        device: str | torch.device = "cuda",
        cache_size: int = 32,
        return_model: bool = False,
        seed: int | None = 42,
    ) -> None:
        super().__init__()

        self.device = torch.device(device)
        self.return_model = return_model
        self._max_cache = cache_size
        self.seed = seed

        (
            _args,
            self.hypermod,
            self.base_model,
            _bm_tok,
            self.emb_model,
            self.emb_tok,
            self.desc_fmt,
            self.pool,
        ) = load_hypermod_checkpoint(Path(hypermod_dir) / "hypermod.pt", self.device)

        self.tokenizer = _bm_tok

        self.hypermod.eval().requires_grad_(False)
        self.base_model.eval()

        cfg_json = Path(hypermod_dir) / "adapter_config.json"
        self._peft_cfg_template: PeftConfig = get_peft_config(
            PeftConfig.from_json_file(cfg_json)
        )
        # expose read-only copy for callers
        self.peft_cfg: PeftConfig = self._peft_cfg_template
        self.base_model_id = self.peft_cfg.base_model_name_or_path

        # wrap the base model once but hand PEFT its own copy so that
        # mutations never propagate back to the template
        self.peft_model = get_peft_model(
            self.base_model, copy.deepcopy(self._peft_cfg_template)
        )

        # Pre‑compute layer indices for the base model once
        self._layer_idx = torch.arange(
            len(get_layers(self.base_model)), dtype=torch.long, device=self.device
        )

        self._cache: Dict[str, Dict[str, Any]] = {}
        self._cache_lock = threading.Lock()

    @torch.inference_mode()
    def _adapter_for(self, task: str) -> Dict[str, Any]:
        """Generate (or retrieve) the LoRA adapter for task."""

        with self._cache_lock:
            if task in self._cache:
                self._cache[task] = self._cache.pop(task)
                return self._cache[task]

        if self.seed is not None:
            torch.manual_seed(self.seed)
            if torch.cuda.is_available() and self.device.type == "cuda":
                torch.cuda.manual_seed_all(self.seed)

        # embed task description
        task_emb = embed_texts(
            [task],
            self.emb_model,
            self.emb_tok,
            self.desc_fmt,
            self.pool,
            device=self.device,
        )

        # project embedding via optional task encoder to match hyper-network's latent space.
        if getattr(self.hypermod, "task_encoder", None) is not None:
            task_emb = self.hypermod.task_encoder(task_emb)["encoded_task_emb"]

        # lora state-dict
        lora_sd = self.hypermod.gen_lora(self._layer_idx, task_emb)

        bundle = {
            "state_dict": lora_sd,
            "config": copy.deepcopy(self._peft_cfg_template),
        }

        with self._cache_lock:
            if len(self._cache) >= self._max_cache:
                self._cache.pop(next(iter(self._cache)))
            self._cache[task] = bundle
        return bundle

    def load_adapter(self, task: str, name: str):
        """Generate the adapter for task and register it in peft_model.

        If name already exists inside self.peft_model we simply
        activate it; otherwise the freshly-generated weights are loaded
        under that name.
        """

        # if present do cheap context switch
        if name in self.peft_model.peft_config:
            self.peft_model.set_adapter(name)
            return name

        bundle = self._adapter_for(task)
        self.apply_bundle(bundle, name=name)
        return name

    def apply_bundle(self, bundle: Dict[str, Any], name: str = "default"):
        """Load the given adapter bundle into the internal PEFT model and activate it."""
        peft_config = bundle["config"]
        if not isinstance(peft_config, PeftConfig):
            peft_config = PeftConfig.from_dict(peft_config)

        if name not in self.peft_model.peft_config:
            self.peft_model.add_adapter(name, peft_config)

        # Load weights for the specified adapter and activate it
        set_peft_model_state_dict(
            self.peft_model, bundle["state_dict"], adapter_name=name
        )
        self.peft_model.set_adapter(name)
        return self.peft_model

    def set_adapter(self, name: str):
        """Activate an already-loaded adapter by name."""
        self.peft_model.set_adapter(name)

    def export(self, task: str, dir_path: Path):
        """saves lora adapter for given task."""
        dir_path = Path(dir_path)
        dir_path.mkdir(parents=True, exist_ok=True)
        bundle = self._adapter_for(task)
        torch.save(bundle["state_dict"], dir_path / "adapter_model.bin")
        bundle["config"].save_pretrained(dir_path)

    @torch.inference_mode()
    def forward(self, task_string: str):
        """Return a LoRA adapter or model specialised for task_string."""
        bundle = self._adapter_for(task_string)

        if not self.return_model:
            return bundle

        adapter_name = f"tmp_{len(self.peft_model.peft_config)}"
        self.load_adapter(task_string, adapter_name)
        return self.peft_model
