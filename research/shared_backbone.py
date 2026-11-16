"""One full InternVL3 instance, full-root named PEFT adapters and routing."""
from contextlib import contextmanager
from threading import RLock
import json
import os
from pathlib import Path

import torch
from torch import nn
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModel
from safetensors import safe_open

from config import BACKBONE_ID, BACKBONE_REVISION, LORA_TARGETS
from execution_policy import selective_fp32, place_numeric_backbone, wait_for_compute_window


class SharedBackbone(nn.Module):
    def __init__(self, model, *, architecture_test=False):
        super().__init__()
        if isinstance(model, PeftModel):
            raise ValueError("Supply an unwrapped InternVL; load adapters through the shared runtime")
        if not hasattr(model, "language_model") or not hasattr(model.language_model, "model"):
            raise ValueError("Expected full InternVL with language_model.model")
        if getattr(model.config, "model_type", None) != "internvl_chat":
            raise ValueError("Expected an InternVL chat configuration")
        lm = model.config.llm_config
        if getattr(lm, "model_type", None) != "qwen2":
            raise ValueError("InternVL3-2B requires its Qwen2 language component")
        if not architecture_test:
            if (lm.hidden_size, lm.num_hidden_layers) != (1536, 28):
                raise ValueError("Unexpected InternVL3-2B language dimensions")
            if getattr(model.config, "_commit_hash", None) != BACKBONE_REVISION:
                raise ValueError("Cannot verify base revision; use SharedBackbone.load")
        self.architecture_test = architecture_test
        self.model = model.requires_grad_(False).eval()
        self._lock = RLock()

    @classmethod
    def load(cls, device="cpu", dtype="bf16", local_files_only=False, source=None):
        device = torch.device(device)
        if device.type == 'cuda':
            wait_for_compute_window()
        precision = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[dtype]
        if device.type == "cpu":
            precision = torch.float32
        source = source or os.environ.get("FINANCERL_BASE_MODEL")
        identity = None
        if source:
            from base_assets import verify_base
            identity = verify_base(source)
        split = device.type == 'cuda' and dtype == 'fp32' and selective_fp32()
        if device.type == 'cuda':
            wait_for_compute_window()
        model = AutoModel.from_pretrained(
            str(source) if source else BACKBONE_ID, revision=BACKBONE_REVISION, trust_remote_code=True,
            torch_dtype=torch.bfloat16 if split else precision, low_cpu_mem_usage=True, use_flash_attn=False,
            local_files_only=True if source else local_files_only,
        )
        if split:
            place_numeric_backbone(model, device)
        else:
            model.to(device)
        if identity:
            # The local files were matched against the complete pinned snapshot.
            model.config._commit_hash = BACKBONE_REVISION
        runtime = cls(model)
        runtime.source_identity = identity
        runtime.execution_mode = 'selective_fp32_checkpointed' if split else 'uniform'
        return runtime

    @property
    def compute_device(self):
        return next(self.base.language_model.model.layers.parameters()).device

    @property
    def compute_dtype(self):
        return next(self.base.language_model.model.layers.parameters()).dtype

    @property
    def base(self):
        return self.model.get_base_model() if isinstance(self.model, PeftModel) else self.model

    @property
    def hidden_size(self):
        return self.base.config.llm_config.hidden_size

    @property
    def adapter_names(self):
        return set(self.model.peft_config) if isinstance(self.model, PeftModel) else set()

    def add_finance_adapter(self, cfg):
        if cfg.adapter_name in self.adapter_names:
            raise ValueError("Finance adapter is already loaded")
        config = LoraConfig(r=cfg.lora_r, lora_alpha=cfg.lora_alpha,
                            lora_dropout=0.0, target_modules=LORA_TARGETS, bias="none", revision=BACKBONE_REVISION)
        if isinstance(self.model, PeftModel):
            self.model.add_adapter(cfg.adapter_name, config)
        else:
            self.model = get_peft_model(self.model, config, adapter_name=cfg.adapter_name)
        # PEFT otherwise records a machine-specific local source path.
        self.model.peft_config[cfg.adapter_name].base_model_name_or_path = BACKBONE_ID
        self.model.peft_config[cfg.adapter_name].revision = BACKBONE_REVISION
        self.activate(cfg.adapter_name)

    def load_adapter(self, path, name, trainable=False):
        if name in self.adapter_names:
            raise ValueError(f"Adapter {name!r} is already loaded")
        path = Path(path)
        config = LoraConfig.from_pretrained(str(path))
        if config.bias != "none":
            raise ValueError("Shared-domain adapters must not replace shared base biases")
        if not self.architecture_test and config.base_model_name_or_path != BACKBONE_ID:
            raise ValueError("Adapter was not exported against the full InternVL3-2B root")
        if not self.architecture_test and config.revision != BACKBONE_REVISION:
            raise ValueError("Adapter configuration does not pin the project's base revision")
        weights = path / "adapter_model.safetensors"
        if not weights.is_file():
            raise ValueError("Export domain adapters with safe_serialization=True")
        with safe_open(str(weights), framework="pt", device="cpu") as stream:
            keys = list(stream.keys())
        prefixes = tuple(f"base_model.model.{root}." for root in ("language_model", "vision_model", "mlp1"))
        if not keys or any(not key.startswith(prefixes) for key in keys):
            raise ValueError("Adapter keys do not use full InternVL-root paths; migrate and verify the export")
        if isinstance(self.model, PeftModel):
            self.model.load_adapter(str(path), adapter_name=name, is_trainable=trainable)
        else:
            self.model = PeftModel.from_pretrained(self.model, str(path),
                                                 adapter_name=name, is_trainable=trainable)
        self.model.eval()

    def load_domain_adapter(self, adapter_path, name, contract_path):
        """Other teams export a base_contract.json beside their adapter/heads."""
        contract = json.loads(Path(contract_path).read_text())
        expected = {"base_model_id": BACKBONE_ID, "base_revision": BACKBONE_REVISION,
                    "adapter_scope": "full_internvl_root", "adapter_name": name}
        if any(contract.get(k) != v for k, v in expected.items()):
            raise ValueError("Domain adapter base ID, revision, name or wrapping scope is incompatible")
        self.load_adapter(adapter_path, name, trainable=False)

    def activate(self, name):
        if name not in self.adapter_names:
            raise ValueError(f"Adapter {name!r} is not loaded")
        if self.model.active_adapter != name:
            self.model.set_adapter(name)

    @contextmanager
    def using(self, name):
        # Active adapter is mutable state: serialize the whole inference call.
        with self._lock:
            previous = self.model.active_adapter
            self.activate(name)
            try:
                yield
            finally:
                self.activate(previous)

    def finance_hidden_states(self, tokens):
        if tokens.device.type == 'cuda':
            wait_for_compute_window()
        self.activate("finance")
        decoder = self.base.language_model.model
        dtype = self.compute_dtype
        tokens = tokens.to(dtype=dtype)
        return decoder(inputs_embeds=tokens,
                       attention_mask=torch.ones(tokens.shape[:2], dtype=torch.long, device=tokens.device),
                       use_cache=False, return_dict=True).last_hidden_state


class DomainRouter:
    """Explicit domain routing; a future classifier can select this domain key.

    Handlers receive (full PEFT-wrapped InternVL, payload). Missing specialists
    are never simulated. All domains must use full-root adapter paths.
    """
    def __init__(self, shared):
        self.shared = shared
        self.handlers = {}

    def register(self, domain, adapter_name, handler):
        if domain in self.handlers:
            raise ValueError(f"Duplicate domain {domain!r}")
        if adapter_name not in self.shared.adapter_names:
            raise ValueError(f"Load adapter {adapter_name!r} before registering its domain")
        if not callable(handler):
            raise TypeError("Domain handler must be callable")
        self.handlers[domain] = (adapter_name, handler)

    @torch.inference_mode()
    def predict(self, domain, payload):
        if domain not in self.handlers:
            raise ValueError(f"Unknown domain {domain!r}; registered: {sorted(self.handlers)}")
        adapter, handler = self.handlers[domain]
        with self.shared.using(adapter):
            if self.shared.model.training:
                raise RuntimeError("Router requires an eval-mode shared backbone")
            return handler(self.shared.model, payload)
