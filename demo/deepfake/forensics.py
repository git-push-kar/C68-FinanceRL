"""Deepfake preprocessing + forensic telemetry (ported for final/).

Sources (verbatim logic):
  build_transform / compute_fft_features — dataset.py (FFT) + inference.py
  extract_forensic_telemetry / build_telemetry_prompt — inference.py
  parse_reasoning / trim_after_answer — inference.py
  build_grounding_prompt / describe_frequency / regions_from_topk — wrapper.py
"""
from __future__ import annotations

import re
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

from .wrapper import (build_grounding_prompt, describe_frequency,
                      regions_from_topk)


def build_transform(image_size: int = 448):
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.CenterCrop(image_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])


def compute_fft_features(img: Image.Image, target_size: int = 224) -> torch.Tensor:
    """Log-magnitude 2D Fourier spectrum across RGB channels. Output (3,H,W)."""
    img_resized = img.resize((target_size, target_size), Image.BILINEAR)
    img_np = np.array(img_resized, dtype=np.float32) / 255.0
    if img_np.ndim == 2:
        img_np = np.stack([img_np] * 3, axis=-1)
    elif img_np.shape[2] == 4:
        img_np = img_np[:, :, :3]
    img_np = np.transpose(img_np, (2, 0, 1))
    fft = np.fft.fft2(img_np, axes=(-2, -1))
    fft_shift = np.fft.fftshift(fft, axes=(-2, -1))
    magnitude = np.abs(fft_shift)
    log_magnitude = np.log1p(magnitude)
    norm = np.zeros_like(log_magnitude)
    for c in range(log_magnitude.shape[0]):
        c_min, c_max = log_magnitude[c].min(), log_magnitude[c].max()
        norm[c] = (log_magnitude[c] - c_min) / (c_max - c_min + 1e-8)
    return torch.from_numpy(norm).float()


def preprocess_image(img: Image.Image, device: str = "cpu") -> Tuple[torch.Tensor, torch.Tensor]:
    """Returns (pixel_values (1,3,448,448), freq_features (1,3,224,224))."""
    pv = build_transform(448)(img.convert("RGB")).unsqueeze(0)
    freq = compute_fft_features(img, target_size=224).unsqueeze(0)
    return pv.to(device), freq.to(device)


_TAG_RE = re.compile(r"<(fast|planning|reasoning|reflection|conclusion)>(.*?)</\1>",
                     re.DOTALL | re.IGNORECASE)
_ANSWER_RE = re.compile(r"<answer>\s*(real|fake)\s*</answer>", re.IGNORECASE)
_ANSWER_BOOL_RE = re.compile(r"<answer>\s*(true|false)\s*</answer>", re.IGNORECASE)


def parse_reasoning(text: str) -> dict:
    tags = {m.group(1).lower(): m.group(2).strip() for m in _TAG_RE.finditer(text)}
    m = _ANSWER_RE.search(text)
    if m:
        tags["answer"] = m.group(1).lower()
        tags["answer_heuristic"] = False
    else:
        mb = _ANSWER_BOOL_RE.search(text)
        if mb:
            tags["answer"] = "real" if mb.group(1).lower() == "true" else "fake"
            tags["answer_heuristic"] = True
        else:
            tags["answer"] = "unknown"
            tags["answer_heuristic"] = False
    return tags


def trim_after_answer(text: str) -> str:
    text = text.replace("<|im_end|>", "").replace("<|endoftext|>", "").replace("<|im_start|>", "")
    end = text.rfind("</answer>")
    if end != -1:
        return text[:end + len("</answer>")].strip()
    return text.strip()


def extract_forensic_telemetry(cls_pred: int, cls_conf: float,
                               freq_features: Optional[torch.Tensor] = None,
                               attn_weights: Optional[torch.Tensor] = None) -> dict:
    verdict = "FAKE" if cls_pred == 1 else "REAL"
    regions = "Uniform facial distribution"
    if attn_weights is not None:
        flat = attn_weights.squeeze().float()
        topk = torch.topk(flat, k=min(8, flat.numel())).indices.cpu().tolist()
        regions = regions_from_topk(topk)
    spectral_desc = "Natural spectral distribution"
    if freq_features is not None:
        ff = freq_features.squeeze()
        spectral_desc = describe_frequency(ff.unsqueeze(0) if ff.dim() == 2 else ff)
    return {"verdict": verdict, "conf": cls_conf,
            "regions": regions, "spectral_desc": spectral_desc}


def build_telemetry_prompt(telemetry: dict) -> str:
    c = telemetry["conf"]
    return build_grounding_prompt(
        verdict=telemetry["verdict"], conf_str=f"confidence {c:.1%}",
        spectral_desc=telemetry["spectral_desc"], regions_str=telemetry["regions"])
