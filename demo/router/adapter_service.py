"""Shared adapter service: one frozen base, lazily loaded task adapters.

Routing layer (Laya) picks the task; this service loads exactly the weights
that task needs, releases the previous task's adapter, and executes it. Generation is a
token-by-token stream (never a single buffered string) so the API can flush
each token to the client as it arrives, with per-token latency attached.

Two generation backends:
  * real  — transformers generate() + TextIteratorStreamer on a thread,
            on the bare base or base + the single resident LoRA.
  * stub  — deterministic word-split stream (FINAL_STUB_GENERATION=1 or
            model unloadable); verifies streaming shape without GPU.
"""
from __future__ import annotations
from pathlib import Path

import logging
import os
import threading
import time
import traceback
from contextlib import contextmanager
from typing import Any, Dict, Generator, Optional

from router.latency import GenerationReceipt, TokenStat
from router.prompts import build_messages
from router.residency import AdapterResidency, Backend

logger = logging.getLogger(__name__)

_ANOMALY = "anomaly"
_REASONING = "reasoning"
_FINANCE = "finance"
_DEEPFAKE_LLM = "deepfake_llm"
_DEEPFAKE_VISION = "deepfake_vision"


def _assert_attached(caught, name: str) -> None:
    """Refuse silent base-model generation when PEFT could not map weights."""
    missing = [str(w.message)[:160] for w in caught
               if "missing adapter keys" in str(w.message)]
    if missing:
        raise RuntimeError(
            f"adapter {name!r} weights did not attach (key namespace "
            f"mismatch): {missing[0]}... Refusing silent base-model generation.")


class TagStripper:
    """Streaming XML-tag filter for non-forensic tasks.

    The LR-fused base leaks template closers (</question>, </answer>) even
    under a no-XML system prompt. Tags may split across token boundaries, so
    a trailing partial tag (`<` + letters, no `>` yet) is held back until it
    completes or is disproved. Plain-text uses of `<` (e.g. `a < b`) pass
    through: only `<letter...>` shapes are treated as tags.
    """

    _TAG_RE = None

    def __init__(self, max_carry: int = 32) -> None:
        import re
        if TagStripper._TAG_RE is None:
            TagStripper._TAG_RE = re.compile(r"</?[A-Za-z][^<>]*>")
        self._carry = ""
        self._max_carry = max_carry

    def feed(self, text: str) -> str:
        data = self._carry + (text or "")
        self._carry = ""
        # Hold back a trailing partial tag.
        lt = data.rfind("<")
        if lt != -1 and ">" not in data[lt:]:
            tail = data[lt:]
            import re as _re
            if _re.match(r"</?[A-Za-z]*$", tail) and len(tail) <= self._max_carry:
                self._carry = tail
                data = data[:lt]
        return TagStripper._TAG_RE.sub("", data)

    def flush(self) -> str:
        tail, self._carry = self._carry, ""
        return TagStripper._TAG_RE.sub("", tail)


class TagSectioner:
    """Streaming section router for the forensic wrapper's tagged reasoning.

    The trained forensic LM emits
    `<fast>/<planning>/<reasoning>/<reflection>/<conclusion>/<answer>`
    sections. Deliberation (`fast/planning/reasoning/reflection`) streams on
    the ``thinking`` channel (rendered collapsed, ChatGPT-style);
    ``conclusion`` is the human-readable summary and streams on its own
    channel so callers can show it; `<answer>` is the verdict word. Unknown
    tags are dropped (name only) and never leak. Tags may split across token
    boundaries, so a trailing partial tag is held back until it completes or
    is disproved.

    `feed()` / `flush()` return ``[(channel, text), ...]`` with channel in
    ``{"thinking", "conclusion", "answer"}``. Accumulated ``answer`` /
    ``conclusion`` text is exposed for post-stream fallbacks.
    """

    _THINKING = ("fast", "planning", "reasoning", "reflection")

    def __init__(self, default: str = "answer", max_carry: int = 48) -> None:
        import re
        self._re = re.compile(r"</?([A-Za-z][\w.-]*)[^<>]*>")
        self._partial_re = re.compile(r"</?[A-Za-z][\w.-]*$")
        self._default = default
        self._section: Optional[str] = None
        self._carry = ""
        self._max_carry = max_carry
        self.answer_text: list = []
        self.conclusion_text: list = []
        self.thinking_text: list = []

    def _channel(self) -> str:
        if self._section == "conclusion":
            return "conclusion"
        if self._section in self._THINKING:
            return "thinking"
        if self._section == "answer":
            return "answer"
        return self._default

    def _push(self, text: str, out: list) -> None:
        if not text:
            return
        channel = self._channel()
        if self._section is None and not text.strip():
            return  # whitespace between sections — never the answer
        if channel == "thinking":
            self.thinking_text.append(text)
        if channel == "answer":
            self.answer_text.append(text)
        if self._section == "conclusion":
            self.conclusion_text.append(text)
        out.append((channel, text))

    def feed(self, text: str) -> list:
        data = self._carry + (text or "")
        self._carry = ""
        out: list = []
        while data:
            lt = data.find("<")
            if lt == -1:
                self._push(data, out)
                break
            if lt > 0:
                self._push(data[:lt], out)
                data = data[lt:]
            gt = data.find(">")
            if gt == -1:  # partial tag: hold if plausible, else emit "<"
                if len(data) <= self._max_carry and self._partial_re.match(data):
                    self._carry = data
                    break
                self._push("<", out)
                data = data[1:]
                continue
            tag = data[:gt + 1]
            m = self._re.match(tag)
            if m:
                name = m.group(1).lower()
                closing = tag.startswith("</")
                if name in self._THINKING or name in ("conclusion", "answer"):
                    self._section = None if closing else name
                # unknown tags are dropped (name + attrs), never leaked
            else:
                self._push("<", out)  # literal "<" (e.g. "a < b>")
                data = data[1:]
                continue
            data = data[gt + 1:]
        return out

    def flush(self) -> list:
        data, self._carry = self._carry, ""
        if not data:
            return []
        if self._partial_re.match(data):
            return []  # truncated tag at stream end
        out: list = []
        self._push(data, out)
        return out


class BusyError(RuntimeError):
    """Another request holds the model (swap or generation in progress)."""


class _CancelCriteria:
    """StoppingCriteria-compatible: stop generate() when the client leaves."""

    def __init__(self, event: threading.Event) -> None:
        self.event = event

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        return self.event.is_set()


class AdapterService:
    """Executes the task Laya picked, loading only what that task needs.

    Residency (see router/residency.py): nothing loads at start-up; each task
    loads its weights on first use and the previous task's adapter is
    released on a task change. Use `lease(task)` around select + generation
    so a concurrent request can never swap weights under a running stream.
    """

    def __init__(self, base_dir: str,
                 anomaly_dir: str,
                 reasoning_dir: str,
                 deepfake_llm_dir: str,
                 deepfake_vision_dir: str,
                 custom_heads_path: Optional[str] = None,
                 policy: Optional[str] = None,
                 finance_dir: Optional[str] = None) -> None:
        # All adapters target the SAME pristine InternVL3-2B base.
        #   anomaly / reasoning -> LoRA on the shared base, one at a time
        #   deepfake            -> dedicated wrapper (own backbone + heads)
        self.base_dir = base_dir
        self.anomaly_dir = anomaly_dir
        self.reasoning_dir = reasoning_dir
        self.finance_dir = finance_dir
        self.deepfake_llm_dir = deepfake_llm_dir
        self.deepfake_vision_dir = deepfake_vision_dir
        self.custom_heads_path = custom_heads_path
        self._tokenizer = None
        self._stub_backend = _stub_env()
        self.residency = AdapterResidency(
            self._make_stub_backend() if self._stub_backend
            else self._make_real_backend(), policy=policy)

    # -- residency backends -------------------------------------------
    def _make_real_backend(self) -> Backend:
        return Backend(
            load_base=self._load_base,
            unload_base=lambda base: None,  # last reference dropped by caller
            attach_lora=self._attach_lora,
            detach_lora=self._detach_lora,
            load_deepfake=self._load_deepfake,
            unload_deepfake=lambda df: None,
        )

    @staticmethod
    def _make_stub_backend() -> Backend:
        """No weights: residency bookkeeping only (UI/demo/tests)."""
        return Backend(
            load_base=lambda: "stub-base",
            unload_base=lambda base: None,
            attach_lora=lambda base, task: f"stub-peft:{task}",
            detach_lora=lambda model: "stub-base",
            load_deepfake=lambda: ("stub-wrapper", None),
            unload_deepfake=lambda df: None,
        )

    def _lora_dir(self, task: str) -> str:
        path = {_ANOMALY: self.anomaly_dir, _REASONING: self.reasoning_dir,
                _FINANCE: self.finance_dir, "finance_policy": os.getenv("FINAL_FINANCE_POLICY_DIR", str(Path(__file__).resolve().parents[1]/"adapters/finance_policy/finance_lora/finance"))}[task]
        if not path:
            raise RuntimeError(f'No model adapter configured for {task}')
        return path

    def _ensure_tokenizer(self):
        if self._tokenizer is None:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(self.base_dir,
                                                trust_remote_code=True)
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token
            self._tokenizer = tok
        return self._tokenizer

    def _load_base(self):
        """Pristine shared base, no adapter attached."""
        from transformers import AutoModel

        tok = self._ensure_tokenizer()
        base = AutoModel.from_pretrained(
            self.base_dir, trust_remote_code=True,
            torch_dtype="auto", device_map="auto", use_flash_attn=False,
        )
        # Required by InternVLChatModel.generate(): maps <IMG_CONTEXT> slots
        # to vision embeddings (text-only calls pass pixel_values=None).
        try:
            base.img_context_token_id = tok.convert_tokens_to_ids("<IMG_CONTEXT>")
        except Exception as exc:
            logger.warning("could not set img_context_token_id: %s", exc)
        base.eval()
        return base

    def _attach_lora(self, base, task: str):
        import warnings
        from peft import PeftModel
        if task == "finance_policy" or (task == _FINANCE and os.getenv("FINAL_FINANCE_TEXT_DTYPE") == "fp32"):
            import torch
            base._finance_restore_dtype = next(base.language_model.parameters()).dtype
            base._finance_restore_vision_device = next(base.vision_model.parameters()).device
            base.vision_model.cpu()
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            base.language_model.float()

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model = PeftModel.from_pretrained(
                base, self._lora_dir(task), adapter_name=task,
                is_trainable=False)
        _assert_attached(caught, task)
        model.eval()
        return model

    @staticmethod
    def _detach_lora(model):
        """Remove LoRA layers WITHOUT merging: the base stays pristine."""
        base = model.unload()
        if hasattr(base, "_finance_restore_dtype"):
            base.language_model.to(dtype=base._finance_restore_dtype)
            base.vision_model.to(base._finance_restore_vision_device)
            del base._finance_restore_dtype
            del base._finance_restore_vision_device
        for attr in ("peft_config", "_hf_peft_config_loaded"):
            if hasattr(base, attr):
                try:
                    delattr(base, attr)
                except Exception:
                    pass
        return base

    def _load_deepfake(self):
        """Dedicated DeepfakeReasoningModel wrapper (own backbone instance).

        Loads the ORIGINAL submodule-relative LoRA safetensors + the 59
        custom-head tensors. Any unmapped key raises — never silent.
        """
        import torch
        from deepfake.loader import load_trained_wrapper

        if not self.custom_heads_path:
            raise RuntimeError(
                "deepfake custom_heads path is required (FINAL_CUSTOM_HEADS)")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return load_trained_wrapper(
            base_dir=self.base_dir,
            llm_adapter_path=self.deepfake_llm_dir,
            vision_adapter_path=self.deepfake_vision_dir,
            heads_path=self.custom_heads_path,
            device=device,
        )

    # -- adapter selection --------------------------------------------
    @staticmethod
    def adapter_names(task: str) -> list:
        return {"anomaly": [_ANOMALY],
                "finance": [_FINANCE],
                "finance_policy": ["Nifty trading adapter"],
                "reasoning": [_REASONING],
                "deepfake": [_DEEPFAKE_LLM, _DEEPFAKE_VISION, "custom_heads"],
                "general": ["base"]}.get(task, [])

    def _select_locked(self, task: str) -> Dict[str, Any]:
        t0 = time.perf_counter()
        swap = self.residency.activate(task)  # ValueError on unknown task
        return {"task": task, "status": 200,
                "adapters": self.adapter_names(task),
                "mode": "stub" if self._stub_backend else "real",
                "swap": swap.to_dict(),
                "latency_ms": round((time.perf_counter() - t0) * 1000.0, 3)}

    def select(self, task: str) -> Dict[str, Any]:
        """Activate `task` (loads/evicts as needed). Returns a receipt.

        Standalone use (scripts). Servers should use `lease()` so the swap
        and the generation that follows are one critical section.
        """
        with self.residency.lock:
            return self._select_locked(task)

    @contextmanager
    def lease(self, task: str, timeout: Optional[float] = None):
        """Hold the model for one request: swap to `task`, then generate.

        Raises BusyError if another request keeps the model longer than
        `timeout` seconds (FINAL_LEASE_TIMEOUT, default 600).
        """
        if timeout is None:
            timeout = float(os.getenv("FINAL_LEASE_TIMEOUT", "600"))
        if not self.residency.lock.acquire(timeout=timeout):
            raise BusyError("model busy with another request")
        try:
            yield self._select_locked(task)
        finally:
            self.residency.lock.release()

    def status(self) -> Dict[str, Any]:
        return {**self.residency.status(),
                "mode": "stub" if self._stub_backend else "real"}

    # -- deepfake forensic analysis (wrapper path) -------------------------
    def analyze_image_stream(self, image_bytes: bytes, max_new_tokens: int = 512,
                             temperature: float = 0.0,
                             repetition_penalty: float = 1.05,
                             stub: Optional[bool] = None) -> Generator[Dict[str, Any], None, None]:
        """Full forensic pass: classifier verdict first, then streamed reasoning.

        Requires `select("deepfake")` / `lease("deepfake")` first. The trained
        forensic LM emits <fast>/<planning>/<reasoning>/<reflection>/
        <conclusion>/<answer> tags; deliberation streams on "thinking", the
        conclusion + verdict on "token".

        Event shapes:
          {"event": "verdict", "verdict": "REAL"|"FAKE", "confidence": f,
           "hotspots": str, "spectral": str, "telemetry_ms": ms}
          {"event": "thinking", ...}  (fast/planning/reasoning/reflection)
          {"event": "token", ...}     (<conclusion> summary + <answer> verdict)
          {"event": "generation_receipt", ...}
        """
        import io as _io
        use_stub = (_stub_env() or self._stub_backend) if stub is None else stub
        if use_stub:
            yield {"event": "verdict", "verdict": "REAL", "confidence": 0.5,
                   "hotspots": "stub", "spectral": "stub",
                   "telemetry_ms": 0.0, "mode": "stub"}
            yield from self._stub_stream("stub forensic reasoning", max_new_tokens)
            return
        import torch
        from PIL import Image as _Image
        from transformers import TextIteratorStreamer

        from deepfake.forensics import (build_telemetry_prompt,
                                        extract_forensic_telemetry,
                                        preprocess_image)
        if self.residency.deepfake is None:
            raise RuntimeError("deepfake wrapper not resident; "
                               "call select('deepfake') first")
        wrapper, tokenizer = self.residency.deepfake
        device = next(wrapper.parameters()).device
        try:
            img = _Image.open(_io.BytesIO(image_bytes)).convert("RGB")
            img.load()
        except Exception:
            yield {"event": "notice",
                   "text": "That file doesn't look like an image I can read. "
                           "Please attach a JPG or PNG face photo and ask again."}
            return
        pv, freq = preprocess_image(img, device=str(device))

        t_tele = time.perf_counter()
        with torch.no_grad():
            _vis = wrapper._get_vision_features(pv, return_attn_weights=True)
            cls_token, pooled, _patches, attn_w = _vis
            freq_emb = wrapper.freq_branch(freq.to(device))
            _dt = wrapper.cls_head.net[0].weight.dtype
            vis_cat = torch.cat([cls_token.to(_dt), pooled.to(_dt),
                                 freq_emb.to(_dt)], dim=-1)
            logits = wrapper.cls_head(vis_cat)
            probs = torch.softmax(logits.float(), dim=-1)
            pred = int(logits.argmax(-1).item())
            conf = float(probs[0, pred].item())
        telemetry = extract_forensic_telemetry(pred, conf, freq, attn_w)
        telemetry_ms = (time.perf_counter() - t_tele) * 1000.0
        yield {"event": "verdict", "verdict": telemetry["verdict"],
               "confidence": round(conf, 4),
               "hotspots": telemetry["regions"],
               "spectral": telemetry["spectral_desc"],
               "telemetry_ms": round(telemetry_ms, 1)}
        prompt_text = build_telemetry_prompt(telemetry)

        # Reasoning generation with token streaming (mirrors predict()).
        with torch.no_grad():
            vis_tokens = wrapper._get_native_visual_tokens(pv)
        inputs_embeds = vis_tokens.to(torch.bfloat16)
        prompt_ids = tokenizer(prompt_text, return_tensors="pt",
                               add_special_tokens=False).input_ids.to(device)
        prompt_embeds = wrapper.backbone.language_model.get_input_embeddings()(prompt_ids)
        inputs_embeds = torch.cat([inputs_embeds,
                                   prompt_embeds.to(inputs_embeds.dtype)], dim=1)
        attention_mask = torch.ones(inputs_embeds.shape[:2], dtype=torch.long,
                                    device=device)
        cancel = threading.Event()
        criteria = [_CancelCriteria(cancel)]
        try:
            stop_ids = tokenizer("</answer>", add_special_tokens=False).input_ids

            def _stop_on_answer(input_ids, scores, **kwargs):
                return input_ids[0, -len(stop_ids):].tolist() == stop_ids

            criteria.append(_stop_on_answer)
        except Exception:
            pass
        streamer = TextIteratorStreamer(tokenizer, skip_special_tokens=True)
        gen_kwargs: Dict[str, Any] = dict(
            inputs_embeds=inputs_embeds, attention_mask=attention_mask,
            streamer=streamer, max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            repetition_penalty=repetition_penalty, use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            stopping_criteria=_stopping_list(criteria),
        )
        if temperature > 0:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = 0.9
        t0 = time.perf_counter()
        thread = threading.Thread(
            target=lambda: self._generate_guarded(
                wrapper.backbone.language_model.generate, gen_kwargs, streamer),
            daemon=True)
        thread.start()
        prefill_ms = (time.perf_counter() - t0) * 1000.0
        per_token: list = []
        idx = 0
        sectioner = TagSectioner(default="answer")
        visible: list = []
        prev = time.perf_counter()
        try:
            for chunk in streamer:
                now = time.perf_counter()
                dt = (now - prev) * 1000.0
                prev = now
                per_token.append(dt)
                idx += 1
                for channel, text in sectioner.feed(chunk):
                    if channel != "thinking":
                        visible.append(text)
                    yield {"event": "thinking" if channel == "thinking" else "token",
                           "index": idx, "text": text,
                           "latency_ms": round(dt, 3),
                           "elapsed_ms": round((now - t0) * 1000.0, 3)}
        finally:
            # Client gone or stream done: stop generate() before the lease
            # is released, so no swap can happen under a live decode.
            cancel.set()
            thread.join(timeout=60)
        elapsed = round((time.perf_counter() - t0) * 1000.0, 3)
        for channel, text in sectioner.flush():  # release any partial-tag carry
            if channel != "thinking":
                visible.append(text)
            yield {"event": "thinking" if channel == "thinking" else "token",
                   "index": idx, "text": text, "latency_ms": 0.0,
                   "elapsed_ms": elapsed}
        # If the run stopped before <conclusion>/<answer> (budget/stop),
        # surface the deliberation so the client never shows an empty reply.
        if not "".join(visible).strip():
            fallback = ("".join(sectioner.conclusion_text) or
                        "".join(sectioner.thinking_text)).strip()
            if fallback:
                yield {"event": "token", "index": idx, "text": fallback,
                       "latency_ms": 0.0, "elapsed_ms": elapsed}
        if gen_kwargs.get("_worker_error"):
            yield {"event": "notice",
                   "text": "Reasoning hit an error partway; verdict above stands. "
                           f"Detail: {gen_kwargs['_worker_error'][:200]}"}
        total = (time.perf_counter() - t0) * 1000.0
        decode_ms = max(total - prefill_ms, 0.0)
        yield {"event": "generation_receipt",
               **GenerationReceipt(prefill_ms=prefill_ms, decode_ms=decode_ms,
                                   tokens=idx,
                                   tok_per_s=idx / max(decode_ms / 1000.0, 1e-9),
                                   per_token_ms=per_token).to_dict()}

    @staticmethod
    def _generate_guarded(generate_fn, gen_kwargs: Dict[str, Any],
                          streamer: Any) -> None:
        """Run generate() off-thread; never leave the SSE iterator hanging.

        On worker error the exception is logged and the streamer is ended so
        the consumer sees ... + generation_receipt(partial) + done instead of
        blocking forever.
        """
        kwargs = {k: v for k, v in gen_kwargs.items() if not k.startswith("_")}
        try:
            generate_fn(**kwargs)
        except Exception as exc:
            logger.warning("generation failed, ending stream: %s\n%s",
                           exc, traceback.format_exc())
            gen_kwargs["_worker_error"] = str(exc)
        finally:
            try:
                streamer.end()
            except Exception:
                pass

    # -- streaming generation ------------------------------------------
    def stream(self, prompt: str, max_new_tokens: int = 256,
               temperature: float = 0.0, repetition_penalty: float = 1.0,
               stub: Optional[bool] = None) -> Generator[Dict[str, Any], None, None]:
        """Yield token events then a final generation receipt.

        Runs on whatever `select()` / `lease()` made resident: the bare base
        for `general`, base + one LoRA for `anomaly` / `reasoning` / `finance`.

        Event shapes:
          {"event": "token", "index": i, "text": tok,
           "latency_ms": ms-for-this-token, "elapsed_ms": ms-since-start}
          {"event": "generation_receipt", **GenerationReceipt.to_dict()}
        """
        use_stub = (_stub_env() or self._stub_backend) if stub is None else stub
        if use_stub or self.residency.runner() is None:
            if not use_stub:
                logger.warning("no model resident (select() not called?); stub stream")
            yield from self._stub_stream(prompt, max_new_tokens)
            return
        yield from self._real_stream(prompt, max_new_tokens, temperature,
                                     repetition_penalty)

    def format_prompt(self, task: str, history: list, message: str) -> str:
        """Training-faithful prompt via the base tokenizer's own template.

        The adapters were trained under <|im_start|>system framing —
        reproduce it here. Falls back to manual ChatML framing when the
        tokenizer is unavailable (stub mode) or has no template.
        """
        messages = build_messages(task, history or [], message)
        try:
            if self._stub_backend:
                raise RuntimeError("stub mode: no tokenizer")
            return self._ensure_tokenizer().apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
        except Exception as exc:
            if not self._stub_backend:
                logger.warning("chat template failed, manual framing: %s", exc)
            parts = []
            for m in messages:
                parts.append(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n")
            parts.append("<|im_start|>assistant\n")
            return "".join(parts)

    def eos_ids(self) -> list:
        tok = self._ensure_tokenizer()
        ids = {tok.eos_token_id}
        try:
            extra = tok.convert_tokens_to_ids("<|im_end|>")
            if isinstance(extra, int) and extra > 0:
                ids.add(extra)
        except Exception:
            pass
        return [i for i in ids if i]

    def _stub_stream(self, prompt: str,
                     max_new_tokens: int) -> Generator[Dict[str, Any], None, None]:
        words = ("Analysis of the request shows task-relevant evidence. "
                 "Key indicators are consistent with the routed specialist. "
                 "See routing receipt for confidence and per-step latency.").split()
        words = words[: max(1, min(max_new_tokens, len(words)))]
        t0 = time.perf_counter()
        per_token = []
        for i, word in enumerate(words):
            step0 = time.perf_counter()
            text = (" " if i else "") + word
            dt = (time.perf_counter() - step0) * 1000.0
            per_token.append(dt)
            yield {"event": "token", "index": i, "text": text,
                   "latency_ms": round(dt, 3),
                   "elapsed_ms": round((time.perf_counter() - t0) * 1000.0, 3)}
        total = (time.perf_counter() - t0) * 1000.0
        yield {"event": "generation_receipt",
               **GenerationReceipt(prefill_ms=0.0, decode_ms=total,
                                   tokens=len(words),
                                   tok_per_s=len(words) / max(total / 1000.0, 1e-9),
                                   per_token_ms=per_token).to_dict()}

    def _real_stream(self, prompt: str, max_new_tokens: int,
                     temperature: float,
                     repetition_penalty: float = 1.0) -> Generator[Dict[str, Any], None, None]:
        from transformers import TextIteratorStreamer

        tok = self._ensure_tokenizer()
        model = self.residency.runner()
        enc = tok(prompt, return_tensors="pt")
        input_device = model.get_input_embeddings().weight.device
        input_ids = enc["input_ids"].to(input_device)
        attention_mask = enc.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(input_device)
        t0 = time.perf_counter()
        cancel = threading.Event()
        streamer = TextIteratorStreamer(tok, skip_prompt=False,
                                        skip_special_tokens=True)
        gen_kwargs: Dict[str, Any] = dict(
            input_ids=input_ids, attention_mask=attention_mask,
            streamer=streamer,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            repetition_penalty=repetition_penalty,
            eos_token_id=self.eos_ids(),
            pad_token_id=tok.pad_token_id,
            stopping_criteria=_stopping_list([_CancelCriteria(cancel)]),
        )
        if temperature > 0:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = 0.9
        thread = threading.Thread(
            target=lambda: self._generate_guarded(model.generate, gen_kwargs,
                                                  streamer),
            daemon=True)
        thread.start()
        prefill_ms = (time.perf_counter() - t0) * 1000.0
        per_token: list = []
        idx = 0
        # First yielded chunk includes prefill; per-token timing from 2nd chunk.
        prev = time.perf_counter()
        try:
            for chunk in streamer:
                now = time.perf_counter()
                dt = (now - prev) * 1000.0
                prev = now
                per_token.append(dt)
                yield {"event": "token", "index": idx, "text": chunk,
                       "latency_ms": round(dt, 3),
                       "elapsed_ms": round((now - t0) * 1000.0, 3)}
                idx += 1
        finally:
            cancel.set()  # client disconnected / done: stop decoding now
            thread.join(timeout=60)
        if gen_kwargs.get("_worker_error"):
            yield {"event": "notice",
                   "text": "Generation hit an error partway; showing what arrived. "
                           f"Detail: {gen_kwargs['_worker_error'][:200]}"}
        total = (time.perf_counter() - t0) * 1000.0
        decode_ms = max(total - prefill_ms, 0.0)
        yield {"event": "generation_receipt",
               **GenerationReceipt(prefill_ms=prefill_ms, decode_ms=decode_ms,
                                   tokens=idx,
                                   tok_per_s=idx / max(decode_ms / 1000.0, 1e-9),
                                   per_token_ms=per_token).to_dict()}


def _stub_env() -> bool:
    return os.getenv("FINAL_STUB_GENERATION", "0") == "1"


def _stopping_list(criteria: list):
    from transformers import StoppingCriteriaList
    return StoppingCriteriaList(criteria)
