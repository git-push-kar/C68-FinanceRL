"""Laya-routed API: one decision model, lazily loaded adapters, streamed answers.

Laya is the sole router (the rule-based static router was removed by
design). It picks either a specialist adapter or the plain base model, and
the AdapterService loads exactly that adapter for the request, releasing
the previous task's adapter (see router/residency.py).

Endpoints:
  GET  /                      — web UI (Chat · Anomaly detection · Finance)
  POST /chat/stream           — end-user chat (Laya-routed) -> SSE
  POST /route/laya/predict    — research endpoint with full receipts -> SSE
  GET  /adapters/status       — what is resident right now + recent swaps
  POST /adapters/release      — unload everything (free GPU memory)
  GET  /anomaly/scenarios     — TEP test-set runs available for simulation
  POST /anomaly/run           — replay one run through the detector -> JSON
  POST /anomaly/reason        — RCA report / follow-up on an event -> SSE
  GET  /route/stats           — rolling Laya routing counters (research)
  GET  /health                — service health

SSE stream shape (text/event-stream, one JSON per `data:` line):
  {"event": "routing_receipt", ...}            # /route/laya/predict only
  {"event": "route", "task": ...}              # chat/anomaly: Laya's pick
  {"event": "adapter", "task", "adapters", "loaded", "unloaded", "cold"}
  {"event": "verdict", ...}                    # deepfake classifier
  {"event": "thinking", "text": tok}           # collapsed reasoning
  {"event": "token", "text": tok}              # visible answer, as it arrives
  {"event": "notice", "text": ...}             # UI feedback, not dialogue
  {"event": "generation_receipt", ...}         # research endpoint only
  {"event": "done"}
"""
from __future__ import annotations

import base64
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional

import anyio

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from router import AdapterService, LayaRouter
from router.adapter_service import BusyError, TagSectioner, TagStripper
from router.prompts import clean_adapter_directive

logger = logging.getLogger(__name__)
app = FastAPI(title="Polyvalent Laya Service", version="0.3.0")
from api.finance import router as finance_router
app.include_router(finance_router)
from api.paper import router as paper_router
app.include_router(paper_router)

_REPO = Path(__file__).resolve().parent.parent


@app.get('/ui/finance.html', include_in_schema=False)
@app.get('/ui/paper.html', include_in_schema=False)
def retired_finance_page():
    return RedirectResponse('/#finance')


app.mount("/ui", StaticFiles(directory=str(_REPO / "ui"), html=True),
          name="ui")


@app.get("/", include_in_schema=False)
def root():
    return FileResponse(str(_REPO / "ui" / "index.html"))


@app.get("/chat", include_in_schema=False)
def chat_page():
    return RedirectResponse(url="/#chat")


_LAYA: Optional[LayaRouter] = None
_SERVICE: Optional[AdapterService] = None

_STATS: Dict[str, Dict[str, float]] = {
    "laya": {"n": 0, "routing_ms": 0.0},
}

TASK_LABELS = {
    "general": "Base model",
    "reasoning": "Logical-reasoning adapter",
    "deepfake": "Deepfake forensics adapter",
    "anomaly": "TEP anomaly adapter",
    "finance": "Finance explanation adapter",
    "finance_policy": "Nifty trading adapter",
    "finance_policy": "Nifty trading adapter",
}


def get_laya() -> LayaRouter:
    global _LAYA
    if _LAYA is None:
        _LAYA = LayaRouter()
    return _LAYA


# Weights live inside this repository (copied from the training runs, which
# keep their originals):
#   models/InternVL3-2B   pristine shared base
#   adapters/anomaly      TEP RCA LoRA (r=16 / a=32)
#   adapters/reasoning    reasoning LoRA (r=64 / a=128)
#   adapters/deepfake     llm_adapter/, vision_adapter/, custom_heads/
_PRISTINE_BASE = _REPO / "models" / "InternVL3-2B"
_ANOMALY_DIR = _REPO / "adapters" / "anomaly"
_REASONING_DIR = _REPO / "adapters" / "reasoning"
_DEEPFAKE_ADAPTERS = _REPO / "adapters" / "deepfake"


def _weights(env: str, local: Path) -> str:
    """Resolve one weight path: the FINAL_* override if it exists, else `local`.

    An override left over from another machine (say FINAL_BASE_DIR pointing at
    a drive that is no longer mounted) used to fail the first routed request
    with an HFValidationError, because transformers reads a non-existent path
    as a Hub repo id. An override that is not there is now logged and skipped,
    and the copy bundled in this repository is used instead.
    """
    configured = (os.getenv(env) or "").strip().strip('"')
    if configured:
        candidate = Path(configured).expanduser()
        if not candidate.is_absolute():
            candidate = _REPO / candidate
        if candidate.exists():
            return str(candidate)
        logger.warning("%s=%s does not exist; falling back to %s",
                       env, configured, local)
    return str(local)


def get_service() -> AdapterService:
    """One service, one pristine base, task adapters loaded on demand.

    Defaults are the in-repo copies above; FINAL_* variables still override
    them when they point at weights that exist. Nothing is loaded here: each
    adapter loads on its first routed request.
    """
    global _SERVICE
    if _SERVICE is None:
        _SERVICE = AdapterService(
            base_dir=_weights("FINAL_BASE_DIR", _PRISTINE_BASE),
            finance_dir=_weights("FINAL_FINANCE_DIR", _REPO / "adapters" / "finance"),
            anomaly_dir=_weights("FINAL_ANOMALY_DIR", _ANOMALY_DIR),
            reasoning_dir=_weights("FINAL_REASONING_DIR", _REASONING_DIR),
            deepfake_llm_dir=_weights("FINAL_DEEPFAKE_LLM_DIR",
                                      _DEEPFAKE_ADAPTERS / "llm_adapter"),
            deepfake_vision_dir=_weights("FINAL_DEEPFAKE_VISION_DIR",
                                         _DEEPFAKE_ADAPTERS / "vision_adapter"),
            custom_heads_path=_weights(
                "FINAL_CUSTOM_HEADS",
                _DEEPFAKE_ADAPTERS / "custom_heads" / "custom_heads.safetensors"),
        )
    return _SERVICE


def _sse(event: Dict[str, Any]) -> str:
    return "data: " + json.dumps(event) + "\n\n"


_END = object()


def _next_or_end(gen):
    try:
        return next(gen)
    except StopIteration:
        return _END


async def _closing(gen) -> AsyncIterator[str]:
    """Drive a sync SSE generator from the threadpool and ALWAYS close it.

    Starlette never closes a sync body iterator when the client disconnects,
    which would leave the generator suspended inside `service.lease()` —
    holding the model lock (and a live generate()) forever. Closing it runs
    the lease/stream `finally` blocks: decoding stops, the lock is released.
    anyio defers cancellation until a running `next()` returns, so `close()`
    never races an executing frame.
    """
    try:
        while True:
            chunk = await anyio.to_thread.run_sync(_next_or_end, gen)
            if chunk is _END:
                break
            yield chunk
    finally:
        with anyio.CancelScope(shield=True):
            await anyio.to_thread.run_sync(gen.close)


class _ClosingStreamingResponse(StreamingResponse):
    async def __call__(self, scope, receive, send) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            aclose = getattr(self.body_iterator, "aclose", None)
            if aclose is not None:
                with anyio.CancelScope(shield=True):
                    await aclose()


def _sse_response(gen) -> StreamingResponse:
    return _ClosingStreamingResponse(_closing(gen),
                                     media_type="text/event-stream")


def _route(payload: Dict[str, Any]):
    decision = get_laya().route(payload)
    _STATS["laya"]["n"] += 1
    _STATS["laya"]["routing_ms"] += decision.latency_ms
    return decision


def _adapter_event(selection: Dict[str, Any]) -> Dict[str, Any]:
    swap = selection.get("swap") or {}
    task = selection.get("task")
    return {"event": "adapter", "task": task,
            "label": TASK_LABELS.get(task, task),
            "adapters": selection.get("adapters", []),
            "loaded": swap.get("loaded", []),
            "unloaded": swap.get("unloaded", []),
            "cold": bool(swap.get("cold")),
            "swap_ms": swap.get("latency_ms"),
            "mode": selection.get("mode")}


_BUSY_TEXT = ("The model is busy with another request — please try again "
              "in a moment.")


def _load_failed_text(task: str, exc: Exception) -> str:
    return (f"I couldn't load the {TASK_LABELS.get(task, task).lower()} on "
            f"this machine ({type(exc).__name__}: {str(exc)[:160]}). Check the "
            "adapter paths in the README and the server log.")


def _stream_text(service: AdapterService, task: str, prompt: str, *,
                 max_new_tokens: int, temperature: float,
                 repetition_penalty: float, stub: Optional[bool],
                 receipts: bool = False,
                 sink: Optional[Dict[str, Any]] = None) -> Iterator[Dict[str, Any]]:
    """Generate on the resident model; yield thinking/token/notice events.

    `reasoning` emits <reasoning>...</reasoning><answer>...</answer>: the
    first section streams as `thinking` (collapsed by clients), the second
    as visible `token`s. Other tasks only need XML-tag stripping.
    """
    sectioner = TagSectioner(default="answer") if task == "reasoning" else None
    scrub = TagStripper()
    saw_visible = False
    for chunk in service.stream(prompt, max_new_tokens=max_new_tokens,
                                temperature=temperature,
                                repetition_penalty=repetition_penalty,
                                stub=stub):
        kind = chunk.get("event")
        if kind == "token":
            raw = chunk.get("text", "")
            meta = ({k: v for k, v in chunk.items() if k not in ("event", "text")}
                    if receipts else {})
            if sectioner is not None:
                for channel, text in sectioner.feed(raw):
                    if not text:
                        continue
                    if channel != "thinking":
                        saw_visible = True
                    yield {**meta, "event": ("thinking" if channel == "thinking"
                                             else "token"), "text": text}
            else:
                text = scrub.feed(raw)
                if text:
                    yield {**meta, "event": "token", "text": text}
        elif kind == "generation_receipt":
            if sink is not None:
                sink["generation"] = chunk
            if receipts:
                yield chunk
        elif kind == "notice":
            yield chunk
    if sectioner is not None:
        for channel, text in sectioner.flush():
            if text:
                if channel != "thinking":
                    saw_visible = True
                yield {"event": ("thinking" if channel == "thinking"
                                 else "token"), "text": text}
        if not saw_visible:
            fallback = "".join(sectioner.thinking_text).strip()
            if fallback:
                yield {"event": "token", "text": fallback}
    else:
        tail = scrub.flush()
        if tail:
            yield {"event": "token", "text": tail}


def _log_run(name: str, record: Dict[str, Any]) -> None:
    try:
        logdir = _REPO / "logs"
        logdir.mkdir(exist_ok=True)
        with open(logdir / name, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({**record, "ts": time.time()}, default=str) + "\n")
    except Exception as exc:  # logging must never break a stream
        logger.warning("run log failed: %s", exc)


def _log_chat_run(record: Dict[str, Any]) -> None:
    _log_run("chat_runs.jsonl", record)


# ---------------------------------------------------------------------
# Research endpoint: full receipts on the wire.
# ---------------------------------------------------------------------

class PredictRequest(BaseModel):
    payload: Dict[str, Any] = Field(default_factory=dict)
    prompt: str = ""
    max_new_tokens: int = 512
    temperature: float = 0.0
    repetition_penalty: float = 1.0  # training-faithful default
    stub: Optional[bool] = None  # force stub stream (research/offline runs)


def _predict_stream(req: PredictRequest):
    service = get_service()
    decision = _route(req.payload)
    yield _sse({"event": "routing_receipt", **decision.to_dict()})
    if decision.abstained:
        yield _sse({"event": "done", "reason": "router-abstained"})
        return
    task = decision.task
    if task == "deepfake" and not req.payload.get("image_base64"):
        # Never load the deepfake wrapper for a request it cannot serve.
        yield _sse({"event": "notice",
                    "text": "Deepfake analysis needs an image — "
                            "attach one and ask again."})
        yield _sse({"event": "done", "reason": "no-image"})
        return
    try:
        with service.lease(task) as selection:
            yield _sse({"event": "selection_receipt", **selection})
            if task == "deepfake":
                for chunk in service.analyze_image_stream(
                        base64.b64decode(req.payload["image_base64"]),
                        max_new_tokens=req.max_new_tokens,
                        temperature=req.temperature,
                        repetition_penalty=req.repetition_penalty,
                        stub=req.stub):
                    yield _sse(chunk)
            else:
                prompt = service.format_prompt(
                    task, [], req.prompt or req.payload.get("text", ""))
                for ev in _stream_text(service, task, prompt,
                                       max_new_tokens=req.max_new_tokens,
                                       temperature=req.temperature,
                                       repetition_penalty=req.repetition_penalty,
                                       stub=req.stub, receipts=True):
                    yield _sse(ev)
    except BusyError:
        yield _sse({"event": "notice", "text": _BUSY_TEXT})
        yield _sse({"event": "done", "reason": "busy"})
        return
    except ValueError as exc:
        yield _sse({"event": "notice", "text": str(exc)})
        yield _sse({"event": "done", "reason": "bad-task"})
        return
    except Exception as exc:
        logger.exception("predict failed for task %s", task)
        yield _sse({"event": "notice", "text": _load_failed_text(task, exc)})
        yield _sse({"event": "done", "reason": "adapter-unavailable"})
        return
    yield _sse({"event": "done"})


@app.post("/route/laya/predict")
def laya_predict(req: PredictRequest):
    return _sse_response(_predict_stream(req))


@app.get("/route/stats")
def stats() -> Dict[str, Any]:
    n = _STATS["laya"]["n"]
    return {"laya": {"n": int(n),
                     "mean_routing_ms": round(
                         _STATS["laya"]["routing_ms"] / max(n, 1), 3)}}


@app.get("/health")
def health() -> Dict[str, Any]:
    return {"status": "ok",
            "stub_generation": os.getenv("FINAL_STUB_GENERATION", "0")}


@app.get("/adapters/status")
def adapters_status() -> Dict[str, Any]:
    return get_service().status()


@app.post("/adapters/release")
def adapters_release() -> Dict[str, Any]:
    service = get_service()
    try:
        with service.residency.lock:
            service.residency.release_all()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return service.status()


# ---------------------------------------------------------------------
# End-user chat: Laya-only routing. Clients see which adapter answered
# (route/adapter events) but no latencies; those go to logs/chat_runs.jsonl.
# ---------------------------------------------------------------------

class ChatTurn(BaseModel):
    role: str = "user"
    content: str = ""


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    history: list = Field(default_factory=list)  # [{role, content}], last 6 used
    image_base64: Optional[str] = None  # attached image -> deepfake path
    max_new_tokens: int = 768  # room for fast+planning+reasoning+conclusion+answer
    temperature: float = 0.0
    repetition_penalty: float = 1.1  # mild loop-breaker for chat
    stub: Optional[bool] = None
    deep_thinking: bool = False  # UI toggle: pin the message to the reasoning (LR) adapter


# Canned notices are UI feedback, not dialogue. Clients (including cached
# older pages) may echo them back in `history`; drop them so the base model
# cannot pattern-match and repeat the notice on the next turn.
_HISTORY_NOISE = (
    "I couldn't tell which specialist",
    "That specialist is still being set up",
    "Please attach the face image",
    "I can check a face image for you",
    "That file doesn't look like an image",
    "Reasoning hit an error",
    "Generation hit an error",
    "The model is busy",
    "I couldn't load the",
)


def _chat_stream(req: ChatRequest):
    service = get_service()
    payload: Dict[str, Any] = {"text": req.message}
    if req.image_base64:
        payload["image_base64"] = req.image_base64

    # Laya only — it is the sole router by design.
    decision = _route(payload)

    # UI toggle: "deep thinking" pins the message to the reasoning (LR)
    # adapter. An attached image still wins — the forensic judge needs pixels.
    if req.deep_thinking and "image_base64" not in payload:
        decision.task = "reasoning"
        decision.confidence = 1.0
        decision.reason = "manual:deep-thinking"
        decision.abstained = False

    log_record: Dict[str, Any] = {"message": req.message,
                                  "routing": decision.to_dict()}

    # The forensic judge needs pixels: ask for the image BEFORE selecting /
    # loading the deepfake wrapper, so text-only requests stay instant and
    # the currently resident adapter is not evicted for nothing.
    if decision.task == "deepfake" and "image_base64" not in payload:
        _log_chat_run({**log_record, "outcome": "no-image"})
        yield _sse({"event": "route", "task": "deepfake",
                    "label": TASK_LABELS["deepfake"]})
        yield _sse({"event": "notice",
                    "text": "I can check a face image for you — attach it with "
                            "the 📎 button and ask again. For anything else, "
                            "just type your question."})
        yield _sse({"event": "done"})
        return

    # Chat UX: an abstention is not a wall. With no specialist signal the
    # plain base model answers; the raw Laya decision stays in the log.
    if decision.task == "needs_clarification":
        log_record["abstain_to_general"] = True
        decision.task = "general"
        decision.confidence = 0.0
        decision.reason = "guard:general"
        decision.abstained = False
    task = decision.task
    yield _sse({"event": "route", "task": task,
                "label": TASK_LABELS.get(task, task)})

    gen: Dict[str, Any] = {}
    try:
        with service.lease(task) as selection:
            log_record["selection"] = selection
            yield _sse(_adapter_event(selection))

            if task == "deepfake":
                for chunk in service.analyze_image_stream(
                        base64.b64decode(payload["image_base64"]),
                        max_new_tokens=req.max_new_tokens,
                        temperature=req.temperature,
                        repetition_penalty=req.repetition_penalty,
                        stub=req.stub):
                    kind = chunk.get("event")
                    if kind == "verdict":
                        log_record["verdict"] = chunk
                        yield _sse(chunk)
                    elif kind in ("thinking", "token", "notice"):
                        yield _sse({"event": kind, "text": chunk.get("text", "")})
                    elif kind == "generation_receipt":
                        gen["generation"] = chunk
            else:
                # Last turns travel as proper chat turns; the prompt is
                # rebuilt in the adapter's training-time framing.
                norm_turns = []
                for t in (req.history or [])[-6:]:
                    if isinstance(t, ChatTurn):
                        t = t.dict()
                    if not (isinstance(t, dict) and t.get("content")):
                        continue
                    role = t.get("role", "user")
                    content = str(t.get("content"))
                    if role == "assistant" and content.startswith(_HISTORY_NOISE):
                        continue  # retry notices must never become context
                    norm_turns.append({"role": role, "content": content})
                prompt_message = (clean_adapter_directive(req.message)
                                  if task == "reasoning" else req.message)
                prompt = service.format_prompt(task, norm_turns, prompt_message)
                for ev in _stream_text(service, task, prompt,
                                       max_new_tokens=req.max_new_tokens,
                                       temperature=req.temperature,
                                       repetition_penalty=req.repetition_penalty,
                                       stub=req.stub, sink=gen):
                    yield _sse(ev)
    except BusyError:
        _log_chat_run({**log_record, "outcome": "busy"})
        yield _sse({"event": "notice", "text": _BUSY_TEXT})
        yield _sse({"event": "done"})
        return
    except Exception as exc:
        logger.exception("chat failed for task %s", task)
        _log_chat_run({**log_record, "outcome": "adapter-unavailable",
                       "error": repr(exc)})
        yield _sse({"event": "notice", "text": _load_failed_text(task, exc)})
        yield _sse({"event": "done"})
        return
    _log_chat_run({**log_record, "outcome": "answered", **gen})
    yield _sse({"event": "done"})


@app.post("/chat/stream")
def chat_stream(req: ChatRequest):
    return _sse_response(_chat_stream(req))


# ---------------------------------------------------------------------
# Anomaly detection tab: replay a TEP test run through the detector, then
# hand the confirmed event's structured evidence to the anomaly adapter
# (routed through Laya like everything else).
# ---------------------------------------------------------------------

@app.get("/anomaly/scenarios")
def anomaly_scenarios() -> Dict[str, Any]:
    from tep_sim import dataset
    scen = dataset.discover()
    root = dataset.dataset_dir()
    return {"dataset_dir": str(root) if root else None,
            "synthetic": all(s.source == "synthetic" for s in scen),
            "detector_backend": ("external" if os.getenv("FINAL_ANOMALY_REPO")
                                 else "builtin"),
            "scenarios": [s.to_dict() for s in scen]}


class AnomalyRunRequest(BaseModel):
    scenario_id: str


@app.post("/anomaly/run")
def anomaly_run(req: AnomalyRunRequest) -> Dict[str, Any]:
    from tep_sim import find, run
    try:
        scenario = find(req.scenario_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="unknown scenario")
    try:
        return run(scenario)
    except Exception as exc:
        logger.exception("anomaly run failed")
        raise HTTPException(status_code=422,
                            detail=f"{type(exc).__name__}: {exc}")


class AnomalyReasonRequest(BaseModel):
    evidence: Dict[str, Any]
    question: Optional[str] = None      # None -> full RCA report (JSON)
    history: List[Dict[str, Any]] = Field(default_factory=list)
    values: Optional[List[float]] = None  # 52-sensor snapshot at the event
    max_new_tokens: int = 768
    temperature: float = 0.0
    repetition_penalty: float = 1.05
    stub: Optional[bool] = None


def _stub_report(ev: Dict[str, Any]) -> str:
    """Stub-mode placeholder report (no model). Clearly marked as such."""
    top = ev.get("top_anomalous_sensors") or []
    return json.dumps({
        "root_cause": "[stub] no model loaded — placeholder diagnosis",
        "affected_subsystem": ev.get("candidate_subsystem", "unknown"),
        "severity": ev.get("severity", "unknown"),
        "confidence": 0.0,
        "summary": "Stub mode is on (FINAL_STUB_GENERATION=1); this report "
                   "is assembled from the evidence, not generated.",
        "reasoning_steps": [f"First onset: {(ev.get('temporal_sequence') or {}).get('sequence', [{}])[0].get('display_name', 'n/a')}",
                            "Mechanism: not generated in stub mode",
                            "Alternatives: not generated in stub mode"],
        "evidence": [f"{s.get('display_name')} observed {s.get('current_value')} "
                     f"vs baseline {s.get('baseline_value')}" for s in top[:3]],
        "reasoning": "n/a (stub)",
        "recommended_action": "Unset FINAL_STUB_GENERATION to get a real diagnosis.",
        "uncertainty": "Placeholder output.",
    }, indent=1)


def _anomaly_stream(req: AnomalyReasonRequest):
    from tep_sim import telemetry_block

    service = get_service()
    ev = req.evidence or {}
    report = not (req.question and req.question.strip())
    block = telemetry_block(ev, for_report=report)
    payload: Dict[str, Any] = {"text": block}
    if req.values and len(req.values) == 52:
        payload["values"] = req.values
    decision = _route(payload)
    task = decision.task
    yield _sse({"event": "route", "task": task,
                "label": TASK_LABELS.get(task, task),
                "reason": decision.reason})
    if task != "anomaly":
        yield _sse({"event": "notice",
                    "text": f"Laya routed this event to '{task}', not the "
                            "anomaly adapter — nothing was generated."})
        yield _sse({"event": "done"})
        return
    if report:
        message = block
        prompt_task, history = "anomaly_report", []
    else:
        message = f"{block}\n\nOperator question: {req.question.strip()}"
        prompt_task, history = "anomaly", [
            t for t in req.history[-6:] if isinstance(t, dict)]
    record = {"event_id": ev.get("event_id"), "question": req.question,
              "routing": decision.to_dict()}
    gen: Dict[str, Any] = {}
    try:
        with service.lease("anomaly") as selection:
            record["selection"] = selection
            yield _sse(_adapter_event(selection))
            use_stub = (req.stub if req.stub is not None
                        else selection.get("mode") == "stub")
            if report and use_stub:
                yield _sse({"event": "token", "text": _stub_report(ev)})
            else:
                prompt = service.format_prompt(prompt_task, history, message)
                for out in _stream_text(service, "anomaly", prompt,
                                        max_new_tokens=req.max_new_tokens,
                                        temperature=req.temperature,
                                        repetition_penalty=req.repetition_penalty,
                                        stub=req.stub, sink=gen):
                    yield _sse(out)
    except BusyError:
        yield _sse({"event": "notice", "text": _BUSY_TEXT})
        yield _sse({"event": "done"})
        return
    except Exception as exc:
        logger.exception("anomaly reasoning failed")
        _log_run("anomaly_runs.jsonl", {**record, "error": repr(exc)})
        yield _sse({"event": "notice", "text": _load_failed_text("anomaly", exc)})
        yield _sse({"event": "done"})
        return
    _log_run("anomaly_runs.jsonl", {**record, **gen})
    yield _sse({"event": "done"})


@app.post("/anomaly/reason")
def anomaly_reason(req: AnomalyReasonRequest):
    return _sse_response(_anomaly_stream(req))
