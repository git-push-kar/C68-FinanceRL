"""Per-step latency calculator for routing + generation comparison research.

Laya and every generation stage records its own step timings so routing cost
and generation cost are attributable at each stage, not just as end-to-end
totals.
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, Generator, List, Optional


@dataclass
class StepRecord:
    name: str
    latency_ms: float


class StepClock:
    """Named-step stopwatch. Use as context manager per step:

        clock = StepClock()
        with clock.step("modality_gate"):
            ...
        receipt = clock.receipt()  # {"steps": {...}, "total_ms": ...}
    """

    def __init__(self) -> None:
        self._steps: List[StepRecord] = []
        self._t0 = time.perf_counter()

    @contextmanager
    def step(self, name: str) -> Generator[None, None, None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            self._steps.append(
                StepRecord(name=name, latency_ms=(time.perf_counter() - start) * 1000.0)
            )

    def record(self, name: str, latency_ms: float) -> None:
        self._steps.append(StepRecord(name=name, latency_ms=latency_ms))

    def total_ms(self) -> float:
        return (time.perf_counter() - self._t0) * 1000.0

    def steps_dict(self) -> Dict[str, float]:
        return {s.name: round(s.latency_ms, 3) for s in self._steps}

    def receipt(self) -> Dict[str, object]:
        return {"steps": self.steps_dict(), "total_ms": round(self.total_ms(), 3)}


@dataclass
class TokenStat:
    """Per-token timing for streamed generation (research comparison)."""
    index: int
    text: str
    latency_ms: float  # time to produce THIS token (decode step)
    elapsed_ms: float  # time since generation start


@dataclass
class GenerationReceipt:
    prefill_ms: float
    decode_ms: float
    tokens: int
    tok_per_s: float
    per_token_ms: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return {
            "prefill_ms": round(self.prefill_ms, 3),
            "decode_ms": round(self.decode_ms, 3),
            "tokens": self.tokens,
            "tok_per_s": round(self.tok_per_s, 2),
            "per_token_ms": [round(v, 3) for v in self.per_token_ms],
        }
