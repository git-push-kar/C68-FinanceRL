"""Router package: Laya (System-1) is the sole router, with per-step latency.

The rule-based StaticRouter was removed by design: routing must be decided by
Laya over either the specialist adapters or the plain base model, so latency
and accuracy numbers stay attributable to one decision model.
"""
from .adapter_service import AdapterService
from .base import TASKS, BaseRouter, RouterDecision
from .latency import GenerationReceipt, StepClock, TokenStat
from .laya_router import LayaRouter
from .prompts import SYSTEM_BY_TASK, build_messages

__all__ = [
    "TASKS",
    "AdapterService",
    "BaseRouter",
    "GenerationReceipt",
    "LayaRouter",
    "RouterDecision",
    "StepClock",
    "SYSTEM_BY_TASK",
    "TokenStat",
    "build_messages",
]
