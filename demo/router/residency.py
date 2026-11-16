"""Adapter residency: what is loaded on the device, and when it changes.

Laya picks a task per request; this module turns that task into exactly the
weights it needs and releases what the previous task needed. Policy:

  * Nothing is loaded at start-up. The first request for a task loads it.
  * At most ONE LoRA (anomaly | reasoning) is attached to the shared base at
    a time. A task change evicts the old LoRA (unwrapped, not merged) before
    the new one is attached, so the base stays pristine.
  * `general` runs the bare shared base: any resident LoRA is evicted.
  * `deepfake` runs its own wrapper (separate backbone + heads). Under the
    default `swap` policy, selecting it releases the shared base, and
    selecting any text task releases the wrapper, so only one backbone is
    ever on the GPU. `keep` keeps backbones cached once loaded (big GPUs);
    LoRAs are still swapped one at a time.

All backend calls (load/unload) are injected so the policy is unit-testable
without torch; `AdapterService` supplies the real PEFT/transformers versions.
"""
from __future__ import annotations

import gc
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

LORA_TASKS = ("anomaly", "reasoning", "finance", "finance_policy")
POLICIES = ("swap", "keep")


def default_policy() -> str:
    p = os.getenv("FINAL_ADAPTER_POLICY", "swap").strip().lower()
    return p if p in POLICIES else "swap"


@dataclass
class Backend:
    """Callables that do the actual (heavy) work."""
    load_base: Callable[[], Any]                      # -> base model
    unload_base: Callable[[Any], None]                # free base
    attach_lora: Callable[[Any, str], Any]            # (base, task) -> peft model
    detach_lora: Callable[[Any], Any]                 # peft model -> base
    load_deepfake: Callable[[], Any]                  # -> (wrapper, tokenizer)
    unload_deepfake: Callable[[Any], None]


@dataclass
class SwapReceipt:
    task: str
    loaded: List[str] = field(default_factory=list)
    unloaded: List[str] = field(default_factory=list)
    reused: List[str] = field(default_factory=list)
    latency_ms: float = 0.0

    @property
    def cold(self) -> bool:
        return bool(self.loaded)

    def to_dict(self) -> Dict[str, Any]:
        return {"task": self.task, "loaded": self.loaded,
                "unloaded": self.unloaded, "reused": self.reused,
                "cold": self.cold, "latency_ms": round(self.latency_ms, 3)}


class AdapterResidency:
    def __init__(self, backend: Backend, policy: Optional[str] = None) -> None:
        self.backend = backend
        self.policy = policy or default_policy()
        self.base: Any = None          # raw shared base (no LoRA wrapper)
        self.model: Any = None         # PeftModel around `base` when a LoRA is on
        self.lora: Optional[str] = None
        self.deepfake: Any = None      # (wrapper, tokenizer)
        self.task: Optional[str] = None
        self.history: List[Dict[str, Any]] = []
        # Serialises swaps AND generation: a swap must never mutate weights
        # under a running generate(). Plain Lock (not RLock) on purpose: SSE
        # generators may be resumed on a different threadpool thread.
        self.lock = threading.Lock()

    # -- public -------------------------------------------------------------
    def activate(self, task: str) -> SwapReceipt:
        """Make `task` runnable. Caller must hold `self.lock`."""
        if task not in ("general", "deepfake") + LORA_TASKS:
            raise ValueError(f"unknown task: {task}")
        t0 = time.perf_counter()
        r = SwapReceipt(task=task)
        if task == "deepfake":
            self._evict_lora(r)
            if self.policy == "swap":
                self._release_base(r)
            self._ensure_deepfake(r)
        else:
            if self.policy == "swap":
                self._release_deepfake(r)
            self._ensure_base(r)
            if task == "general":
                self._evict_lora(r)
            else:
                self._ensure_lora(task, r)
        self.task = task
        r.latency_ms = (time.perf_counter() - t0) * 1000.0
        self.history.append({**r.to_dict(), "ts": time.time()})
        del self.history[:-50]
        if r.loaded or r.unloaded:
            logger.info("adapter swap -> %s: loaded=%s unloaded=%s (%.0f ms)",
                        task, r.loaded, r.unloaded, r.latency_ms)
        return r

    def runner(self) -> Any:
        """Model to call generate() on for the current text task."""
        return self.model if self.model is not None else self.base

    def status(self) -> Dict[str, Any]:
        return {
            "policy": self.policy,
            "task": self.task,
            "base_loaded": self.base is not None,
            "lora": self.lora,
            "deepfake_loaded": self.deepfake is not None,
            "busy": self.lock.locked(),
            "recent_swaps": self.history[-8:],
        }

    def release_all(self) -> None:
        r = SwapReceipt(task="(release)")
        self._evict_lora(r)
        self._release_base(r)
        self._release_deepfake(r)
        self.task = None

    # -- internals ----------------------------------------------------------
    def _ensure_base(self, r: SwapReceipt) -> None:
        if self.base is None:
            self.base = self.backend.load_base()
            r.loaded.append("base")
        else:
            r.reused.append("base")

    def _release_base(self, r: SwapReceipt) -> None:
        if self.base is None:
            return
        self._evict_lora(r)
        self.backend.unload_base(self.base)
        self.base = None
        r.unloaded.append("base")
        _free_memory()

    def _ensure_lora(self, task: str, r: SwapReceipt) -> None:
        if self.lora == task and self.model is not None:
            r.reused.append(task)
            return
        self._evict_lora(r)
        try:
            self.model = self.backend.attach_lora(self.base, task)
        except Exception:
            # Attach failed half-way: the base may carry a broken wrapper.
            # Drop it so the next request reloads a clean base.
            logger.exception("attaching %s failed; dropping shared base", task)
            self.model, self.lora = None, None
            try:
                self.backend.unload_base(self.base)
            finally:
                self.base = None
                _free_memory()
            raise
        self.lora = task
        r.loaded.append(task)

    def _evict_lora(self, r: SwapReceipt) -> None:
        if self.model is None:
            return
        old = self.lora
        self.base = self.backend.detach_lora(self.model)
        self.model, self.lora = None, None
        r.unloaded.append(old or "lora")
        _free_memory()

    def _ensure_deepfake(self, r: SwapReceipt) -> None:
        if self.deepfake is None:
            self.deepfake = self.backend.load_deepfake()
            r.loaded.append("deepfake")
        else:
            r.reused.append("deepfake")

    def _release_deepfake(self, r: SwapReceipt) -> None:
        if self.deepfake is None:
            return
        self.backend.unload_deepfake(self.deepfake)
        self.deepfake = None
        r.unloaded.append("deepfake")
        _free_memory()


def _free_memory() -> None:
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
