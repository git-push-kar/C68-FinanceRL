"""Ported deepfake stack: wrapper (architecture) + forensics + trained loader."""
from .forensics import (build_telemetry_prompt, build_transform,
                        compute_fft_features, extract_forensic_telemetry,
                        parse_reasoning, preprocess_image, trim_after_answer)
from .loader import load_trained_wrapper
from .wrapper import DeepfakeReasoningModel

__all__ = [
    "DeepfakeReasoningModel",
    "build_telemetry_prompt",
    "build_transform",
    "compute_fft_features",
    "extract_forensic_telemetry",
    "load_trained_wrapper",
    "parse_reasoning",
    "preprocess_image",
    "trim_after_answer",
]
