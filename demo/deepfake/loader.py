"""Trained-weight injection for the ported DeepfakeReasoningModel wrapper.

The extracted adapter safetensors use submodule-relative PEFT key space
(no adapter-name infix — see extraction report):
  llm:    base_model.model.model.layers.<N>.<mod>.lora_{A,B}.weight
  vision: base_model.model.encoder.layers.<N>.<mod>.lora_{A,B}.weight
The live wrapper (built by wrapper.py, same as the training harness) holds
PeftModels on backbone.language_model / backbone.vision_model, i.e. live keys
  backbone.language_model.base_model.model.model.layers.<N>.<mod>.lora_{A,B}.default.weight
  backbone.vision_model.base_model.model.encoder.layers.<N>.<mod>.lora_{A,B}.default.weight
Every checkpoint key is remapped ANCHOR-first (no endswith scan): unmapped
keys raise instead of silently dropping — silent non-attachment is what made
the raw-InternVL deepfake path generate base-model text.
custom_heads (59 bare tensors) load 1:1 onto the wrapper's head modules.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Tuple

import torch
from safetensors.torch import load_file as _load_st
from transformers import AutoTokenizer

from .wrapper import DeepfakeReasoningModel

logger = logging.getLogger(__name__)


def _remap_lora_keys(sd: Dict[str, torch.Tensor], live_prefix: str,
                     ckpt_prefix: str) -> Dict[str, torch.Tensor]:
    """ckpt `base_model.model.<rest>.lora_{A,B}.weight`
    -> live `<live_prefix>.base_model.model.<rest>.lora_{A,B}.default.weight`."""
    out: Dict[str, torch.Tensor] = {}
    for k, v in sd.items():
        if not k.startswith(ckpt_prefix):
            raise KeyError(f"unexpected checkpoint key (namespace drift?): {k}")
        rest = k[len(ckpt_prefix):]  # e.g. model.layers.0....lora_A.weight
        if not rest.endswith(".weight"):
            raise KeyError(f"unexpected LoRA key shape: {k}")
        live_key = live_prefix + ".base_model.model." + rest[: -len(".weight")] + ".default.weight"
        out[live_key] = v
    return out


def copy_lora_into_live(wrapper: DeepfakeReasoningModel,
                         ckpt_path: str, live_prefix: str,
                         ckpt_prefix: str) -> int:
    """Copy trained LoRA tensors into the wrapper's live default-adapter slots."""
    sd = _load_st(ckpt_path, device="cpu")
    mapped = _remap_lora_keys(sd, live_prefix, ckpt_prefix)
    live = dict(wrapper.named_parameters())
    missing = [k for k in mapped if k not in live]
    if missing:
        raise RuntimeError(f"{len(missing)} trained LoRA keys have no live target "
                           f"(e.g. {missing[0]}); refusing partial attach.")
    with torch.no_grad():
        for k, v in mapped.items():
            live[k].data.copy_(v.to(live[k].dtype))
    return len(mapped)


def load_trained_wrapper(base_dir: str, llm_adapter_path: str,
                         vision_adapter_path: str, heads_path: str,
                         device: str = "cpu") -> Tuple[DeepfakeReasoningModel, object]:
    """Build wrapper scaffolding + inject ALL trained weights. Returns (model, tokenizer)."""
    logger.info("building DeepfakeReasoningModel scaffolding ...")
    wrapper = DeepfakeReasoningModel(
        model_path=base_dir, arch_version="v2",
        attn_implementation="sdpa", use_gradient_checkpointing=False,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    wrapper.set_tokenizer(tokenizer)

    n_llm = copy_lora_into_live(
        wrapper, str(Path(llm_adapter_path) / "adapter_model.safetensors"),
        live_prefix="backbone.language_model", ckpt_prefix="base_model.model.")
    n_vis = copy_lora_into_live(
        wrapper, str(Path(vision_adapter_path) / "adapter_model.safetensors"),
        live_prefix="backbone.vision_model", ckpt_prefix="base_model.model.")
    logger.info("injected LLM LoRA tensors=%d vision LoRA tensors=%d", n_llm, n_vis)

    heads = _load_st(heads_path, device="cpu")
    missing, unexpected = wrapper.load_state_dict(heads, strict=False)
    missing_heads = [k for k in missing
                     if k.split(".")[0] in ("attn_pool", "cls_head", "freq_branch",
                                             "custom_projector", "domain_router")]
    if missing_heads:
        raise RuntimeError(f"custom heads did not attach: {missing_heads[:5]}")
    logger.info("custom heads attached (%d tensors, missing-nonhead=%d)",
                len(heads), len(missing))
    wrapper.to(device)
    wrapper.eval()
    return wrapper, tokenizer
