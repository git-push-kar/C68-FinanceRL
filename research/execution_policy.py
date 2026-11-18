"""Execution changes only: exact FP32 finance arithmetic and a quiet window."""
from datetime import datetime, timedelta, timezone
import functools
import json
import os
from pathlib import Path
import time

IST = timezone(timedelta(hours=5, minutes=30))


def quiet_now(moment=None):
    moment = moment or datetime.now(IST)
    # Ten minutes to cool before the user's 23:00 boundary.
    minutes = moment.hour * 60 + moment.minute
    return minutes >= 22 * 60 + 50 or minutes < 7 * 60


def wait_for_compute_window():
    announced = False
    while quiet_now():
        if not announced:
            print('QUIET WINDOW: GPU work paused until 07:00 IST.', flush=True)
            announced = True
        time.sleep(20)
    if announced:
        print('DAYTIME WINDOW: resuming GPU work.', flush=True)


def enable_eval_checkpointing(decoder):
    """Preserve eval/dropout behavior while recomputing layer activations.

    Upstream gradient_checkpointing normally requires training=True. PPO needs
    eval-mode likelihoods, so wrap each layer without changing its mode.
    """
    import torch
    from torch.utils.checkpoint import checkpoint
    for layer in decoder.layers:
        if getattr(layer, '_finance_checkpointed', False):
            continue
        original = layer.forward

        def wrap(forward):
            @functools.wraps(forward)
            def call(*args, **kwargs):
                if torch.is_grad_enabled():
                    return checkpoint(forward, *args, use_reentrant=False,
                                      preserve_rng_state=True, **kwargs)
                return forward(*args, **kwargs)
            return call

        layer.forward = wrap(original)
        layer._finance_checkpointed = True


def selective_fp32():
    return os.environ.get('FINANCERL_EXECUTION') == 'selective_fp32'


def place_numeric_backbone(model, device):
    """Keep one full pristine object; move only the used decoder to GPU.

    Source weights are BF16. Converting the active path to FP32 is lossless;
    inactive source tensors retain their original BF16 representation on CPU.
    No quantization, frozen-layer removal or adapter-rank reduction is used.
    """
    import torch
    wait_for_compute_window()
    decoder = model.language_model.model
    decoder.layers.to(device=device, dtype=torch.float32)
    decoder.norm.to(device=device, dtype=torch.float32)
    if hasattr(decoder, 'rotary_emb'):
        decoder.rotary_emb.to(device=device, dtype=torch.float32)
    enable_eval_checkpointing(decoder)

