"""Laya (System-1 decision model) router — the sole router.

Laya decides over either the specialist adapters or the plain base model.
Zero-shot Laya is uncalibrated and occasionally picks a specialist for plain
conversation ("what is your name" -> deepfake), so its choice is passed
through a deterministic support guard: a specialist is only accepted when the
input actually contains cues for it. Anything else falls back to `general`
(the plain chat base), never to a canned clarification wall. The guard is a
safety net around Laya, not a second router: every override is logged with
`reason=guard:*` and the raw Laya probabilities stay in `extra`.

Steps instrumented separately for research:
  1. pre_gate      — free modality check (image/sensor_vector short-circuit;
                     input shaping only)
  2. laya_load     — checkpoint resident check / lazy load (amortized)
  3. laya_forward  — single non-autoregressive forward pass
  4. arbitration   — threshold + support guard
"""
from __future__ import annotations

import os
import re
from typing import Any, Dict, Optional

from .base import BaseRouter, RouterDecision
from .latency import StepClock

# The SDK's abstention gate is defined on `answer_confidence` = max(p), not
# on the entropy `confidence` (see laya/confidence.py): the latter moves with
# the option count and is explicitly not calibrated. This question has four
# options, so uniform = 0.25; 0.30 accepts decisions that are at least a little
# sharper than uniform and abstains on genuinely undecided text. Zero-shot
# Laya is uncalibrated and jitters run-to-run, so raise this via env after
# fine-tuning. Both numbers are logged (logs/chat_runs.jsonl, /route/stats).
# Env override: FINAL_LAYA_MIN_CONF.
LAYA_MIN_CONFIDENCE = float(os.getenv("FINAL_LAYA_MIN_CONF", "0.30"))

_CHOICE_QUESTION = {
    "task": {
        "type": "choice",
        "instructions": "Route this request to the correct specialist adapter.",
        "criteria": {
            "deepfake": "face image, real/fake verdict, spectral or forgery artifacts",
            "anomaly": ("industrial process fault / anomaly diagnosis, root "
                        "cause, TEP (Tennessee Eastman), sensors XMEAS/XMV, "
                        "pressure, temperature, level, valve, reactor, severity"),
            "reasoning": ("formal logic, deduction, syllogisms, logic/ordering "
                          "puzzles, 'if ... then' constraints, proofs, "
                          "step-by-step mathematical or logical derivation"),
            "general": "everyday knowledge, definitions, chit-chat, coding, or any request fitting no specialist",
        },
    }
}

# -- support guard cues ---------------------------------------------------
# A specialist is accepted only if the request has cues for it. These are
# intentionally conservative: plain chat ("hi", "what is your name", "write a
# Python function") must reach `general`, while explicit domain wording still
# reaches the right adapter even when Laya abstains.
_ANOMALY_STRONG = re.compile(
    r"\b(xmeas|xmv|tennessee\s+eastman|tep|stiction|root\s+cause|"
    r"sensor\s+anomal(?:y|ies)|valve\s+stuck|fault\s+diagnos\w*|"
    r"diagnose\s+the\s+fault|reactor\s+(?:pressure|temperature|level))\b",
    re.I)
_ANOMALY_WEAK = re.compile(
    r"\b(reactor|pressure|temperature|level|valve|sensor|fault|anomaly|"
    r"industrial|process|actuator|cooling|stripper|separator|severity|"
    r"plant)\b", re.I)
_ANOMALY_DOMAIN = re.compile(
    r"\b(reactor|valve|sensor|actuator|industrial|plant|fault|anomaly|"
    r"cooling|stripper|separator|severity)\b", re.I)
_IMAGE_CUE = re.compile(
    r"\b(face|photo|image|picture|selfie|portrait)\b", re.I)
_IMAGE_VERDICT = re.compile(
    r"\b(real|fake|forgery|forged|manipulated|authentic|deepfake|synthetic)\b",
    re.I)
_IMAGE_ANALYZE = re.compile(
    r"\b(this|that|attached|uploaded|following|my|the)\b|"
    r"\b(detect|check|analyze|analyse|verify|classify|determine)\b", re.I)
_IMAGE_DEFINITION = re.compile(
    r"^\s*(what|who|define|explain|describe|tell\s+me\s+about)\b", re.I)
_REASONING_CUE = re.compile(
    r"\b(logic|logical|syllogism|deduce|deduction|infer|inference|premise|"
    r"proof|prove|puzzle|constraint|mortal|taller|tallest|therefore|"
    r"equation|algebra|solve|mathematical|math)\b|"
    r"\bif\s+.+\s+then\b|\ball\s+\w+\s+are\b|step[- ]by[- ]step|"
    # A decision question needs an explicit choice set.  Bare "should I ..."
    # advice ("should I carry the hen?") belongs to general chat, not to the
    # logic adapter (which force-fits it to its trained carwash/commute cases).
    r"\bshould\s+(?:i|we)\b[^?.!]*\b(?:or|vs\.?|versus)\b|"
    r"\bwhich\s+(?:one\s+)?should\b|"
    r"\bwhich\s+is\s+better\b[^?.!]*\b(?:or|vs\.?|versus)\b|"
    r"\b(?:choose|pick|decide)\b[^?.!]*\b(?:between|or|vs\.?|versus)\b|"
    r"\b(?:pros?\s+and\s+cons?|trade-?off)\b|"
    r"\bis\s+it\s+better\b[^?.!]*\b(?:or|than)\b|"
    r"\bbetter\s+to\b[^?.!]*\b(?:or|than)\b", re.I)
# User-requested adapter overrides: "load/use/switch to the reasoning adapter".
_EXPLICIT_REASONING = re.compile(
    r"\b(?:use|load|switch\s+to|run|enable|activate|call)\b.{0,24}"
    r"\breasoning\b|\breasoning\s+(?:adapter|model|engine)\b", re.I)


def _explicit_reasoning(text: str) -> bool:
    return bool(_EXPLICIT_REASONING.search(text or ""))


def _text_signals(text: str) -> Dict[str, bool]:
    """Return which specialists the *text* actually asks for."""
    t = text or ""
    strong = bool(_ANOMALY_STRONG.search(t))
    weak = {m.group(1).lower() for m in _ANOMALY_WEAK.finditer(t)}
    anomaly = strong or (len(weak) >= 2 and bool(_ANOMALY_DOMAIN.search(t)))

    image_terms = bool(_IMAGE_CUE.search(t) and _IMAGE_VERDICT.search(t))
    definition = bool(_IMAGE_DEFINITION.search(t)
                      and not re.search(r"\b(this|attached|uploaded)\b", t, re.I))
    deepfake = bool(image_terms and _IMAGE_ANALYZE.search(t)
                    and not definition)

    reasoning = bool(_REASONING_CUE.search(t))
    return {"anomaly": anomaly, "deepfake": deepfake, "reasoning": reasoning}


class LayaRouter(BaseRouter):
    """Sole router. The `laya` package is imported lazily, so a missing
    install degrades to an explicit abstention rather than a crash."""

    name = "laya"

    def __init__(self, device: Optional[str] = None,
                 min_confidence: float = LAYA_MIN_CONFIDENCE) -> None:
        self._router = None
        self._device = device
        self._load_error: Optional[str] = None
        self.min_confidence = min_confidence

    def route(self, payload: Dict[str, Any]) -> RouterDecision:
        clock = StepClock()

        with clock.step("pre_gate"):
            shaped = self._shape(payload)

        with clock.step("laya_load"):
            ok = self._ensure_loaded()

        raw: Dict[str, Any] = {}
        if ok:
            with clock.step("laya_forward"):
                raw = self._router.predict(shaped, _CHOICE_QUESTION)
            with clock.step("arbitration"):
                task, confidence, reason, abstained = self._arbitrate(raw)
        else:
            task, confidence, reason, abstained = (
                "needs_clarification", 0.0,
                f"laya-unavailable:{self._load_error}", True)

        with clock.step("guard"):
            task, confidence, reason, abstained, guard = self._guard(
                shaped, task, confidence, reason, abstained)

        total = clock.total_ms()
        ans = (raw.get("answers") or {}).get("task", {}) or {}
        return RouterDecision(
            router="laya",
            task=task,
            confidence=confidence,
            reason=reason,
            latency_ms=total,
            steps=clock.steps_dict(),
            abstained=abstained,
            extra={
                "laya_available": ok,
                "laya_model": (raw.get("routing") or {}).get("model", "unknown"),
                "laya_answer_confidence": ans.get("answer_confidence"),
                "laya_entropy_confidence": ans.get("confidence"),
                "laya_probabilities": ans.get("probabilities"),
                "guard": guard,
            },
        )

    # -- internals -----------------------------------------------------
    def _ensure_loaded(self) -> bool:
        if self._router is not None:
            return True
        try:
            from . import _torch_shim  # noqa: F401  (eager fallback, Windows-CPU)
            from laya import Router  # lazy: static path never imports this
            kwargs: Dict[str, Any] = {}
            if self._device:
                kwargs["device"] = self._device
            self._router = Router(**kwargs)
            # Only `english` and `multilingual` are reachable: routing is by
            # script/language, and `typed-decisions` is never selected unless
            # auto_task_detection is on. Preloading just the two keeps the cold
            # start to ~743M params instead of ~1.16B (Router(preload=True)).
            self._router.preload(["english", "multilingual"])
            return True
        except Exception as exc:  # laya missing / no weights / no net
            self._load_error = f"{type(exc).__name__}: {exc}"
            return False

    def _shape(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Free input shaping only — modality short-circuits obvious cases
        without spending a forward pass; text goes to Laya untouched."""
        if any(payload.get(k) for k in
               ("image_bytes", "image_base64", "image_path", "pixel_values")):
            return {"modality": "image", "body": "face image for real/fake verdict"}
        values = payload.get("values")
        if isinstance(values, list) and len(values) == 52:
            return {"modality": "sensor_vector",
                    "body": payload.get("text", "TEP 52-channel sensor window")}
        return {"modality": "text",
                "body": payload.get("text") or payload.get("prompt") or ""}

    def _guard(self, shaped: Dict[str, Any], task: str, confidence: float,
               reason: str, abstained: bool) -> tuple:
        """Accept a specialist only when the input actually supports it.

        Returns ``(task, confidence, reason, abstained, guard_tag)``. The
        guard never invents a specialist: unsupported Laya choices and plain
        chat fall back to `general` so the base model can answer.
        """
        modality = shaped.get("modality")
        if modality == "image":
            return ("deepfake", confidence, "guard:image", False, "image")
        if modality == "sensor_vector":
            return ("anomaly", confidence, "guard:sensor_vector", False,
                    "sensor_vector")

        body = shaped.get("body") or ""
        if _explicit_reasoning(body):
            return ("reasoning", confidence, "guard:explicit-reasoning", False,
                    "explicit-reasoning")

        sig = _text_signals(body)
        if task in ("anomaly", "deepfake", "reasoning") and not sig[task]:
            picked = self._pick(sig)
            return (picked, confidence, f"guard:{task}->{picked}", False,
                    "unsupported-specialist")
        if abstained or task in ("general", "needs_clarification"):
            picked = self._pick(sig)
            if picked != "general":
                return (picked, confidence, f"guard:{task}->{picked}", False,
                        "signal-fallback")
            return ("general", confidence, "guard:general", False, "general")
        return (task, confidence, reason, abstained, None)

    @staticmethod
    def _pick(sig: Dict[str, bool]) -> str:
        if sig["anomaly"]:
            return "anomaly"
        if sig["reasoning"]:
            return "reasoning"
        if sig["deepfake"]:
            return "deepfake"
        return "general"

    def _arbitrate(self, raw: Dict[str, Any]) -> tuple:
        try:
            ans = (raw.get("answers") or {}).get("task", {}) or {}
            choice = ans.get("choice", "general")
            # Gate on the SDK-defined abstention quantity: max(p). Fall back
            # to the entropy field only if an older payload omits it.
            conf_raw = ans.get("answer_confidence")
            if not isinstance(conf_raw, (int, float)):
                conf_raw = ans.get("confidence", 0.0)
            conf = float(conf_raw)
        except (ValueError, TypeError, AttributeError):
            return ("needs_clarification", 0.0, "laya-malformed-answer", True)
        if conf < self.min_confidence:
            # No substitution: explicit abstention keeps numbers attributable.
            return ("needs_clarification", conf,
                    f"laya-lowconf:{choice}@{conf:.2f}<{self.min_confidence}", True)
        return (choice, conf, f"laya:{choice}", False)
