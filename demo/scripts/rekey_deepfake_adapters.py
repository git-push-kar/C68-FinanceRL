"""One-time re-key of deepfake LoRA safetensors into top-level namespace.

Checkpoint keys were saved submodule-relative by the deepfake training
harness (PeftModel wrapped backbone.language_model / backbone.vision_model
directly):
  llm:    base_model.model.model.layers...            (missing language_model)
  vision: base_model.model.encoder.layers...          (missing vision_model)

The anomaly adapter (and our single-PeftModel service) uses top-level
paths: base_model.model.language_model.model.layers... /
base_model.model.vision_model.encoder.layers...

This script writes corrected COPIES (originals untouched) + verifies counts.
"""
import sys
from safetensors.torch import load_file, save_file

sys.path.insert(0, r"C:\Users\Admin\Desktop\final")

_Q = (r"C:\Users\Admin\Desktop\final\quarantine"
      r"\2026-10-21_fused_lr_and_ui\adapters\Deepfake_adapter")
JOBS = [
    (_Q + r"\llm_adapter\adapter_model.safetensors",
     _Q + r"\llm_adapter_toplevel\adapter_model.safetensors",
     "base_model.model.model.layers.",
     "base_model.model.language_model.model.layers."),
    (_Q + r"\vision_adapter\adapter_model.safetensors",
     _Q + r"\vision_adapter_toplevel\adapter_model.safetensors",
     "base_model.model.encoder.layers.",
     "base_model.model.vision_model.encoder.layers."),
]

for src, dst, old, new in JOBS:
    sd = load_file(src, device="cpu")
    fixed, untouched = {}, 0
    for k, v in sd.items():
        if k.startswith(old):
            fixed[new + k[len(old):]] = v
        else:
            fixed[k] = v
            untouched += 1
    import os
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    save_file(fixed, dst)
    print(f"{src.split('Deepfake_adapter')[-1]}: {len(sd)} keys -> "
          f"{len(fixed)} keys, untouched(non-matching)={untouched}")
print("REKEY OK")
