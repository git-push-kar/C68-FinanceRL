"""Smoke: Laya routing receipt + per-step latency + token streaming shape.

Laya is the sole router. The rule-based static router was removed by design,
so this smoke no longer asserts a static decision: it checks the receipt
schema and asserts that routing NEVER reports static-style reasons
(`modality=`, `keyword=`) and never silently substitutes a decision.
"""
import json
import os
import sys

sys.path.insert(0, r"C:\Users\Admin\Desktop\final")
os.environ["FINAL_STUB_GENERATION"] = "1"

from router import LayaRouter

cases = {
    "image": {"image_bytes": "AAA", "text": "is this face real?"},
    "sensor": {"values": [0.1] * 52, "text": "XMEAS_7 pressure high, XMV_3 stuck"},
    "general": {"text": "What is the capital of France?"},
    "vague": {"text": "hello there"},
}

laya = LayaRouter()
abstained = 0
for name, payload in cases.items():
    d = laya.route(payload)
    print(f"[laya/{name}]", json.dumps(d.to_dict()))
    assert not d.reason.startswith(("modality=", "keyword=")), \
        "laya must never fall back to static logic"
    if d.abstained:
        abstained += 1
        assert d.reason.startswith("laya-"), d.reason

# Streaming shape (only meaningful when Laya is installed and routes):
# routing_receipt -> selection -> tokens -> generation_receipt -> done
from api.server import _predict_stream, PredictRequest

req = PredictRequest(payload=cases["sensor"], prompt="summarise", max_new_tokens=20)
events = []
for raw in _predict_stream(req):
    assert raw.startswith("data: "), raw[:20]
    events.append(json.loads(raw[len("data: "):]))
kinds = [e["event"] for e in events]
print("stream events:", kinds)
assert kinds[0] == "routing_receipt"
if "routing_receipt" in kinds and events[0].get("abstained"):
    print("Laya unavailable/abstained: routing receipt shape verified; "
          "skipping generation assertions.")
    print("SMOKE OK (abstain path)")
    sys.exit(0)
assert "selection_receipt" in kinds and kinds[-1] == "done"
toks = [e for e in events if e["event"] == "token"]
assert toks and all("latency_ms" in t and "elapsed_ms" in t for t in toks), "per-token latency required"
gen = next(e for e in events if e["event"] == "generation_receipt")
assert gen["tokens"] == len(toks) and "tok_per_s" in gen and "per_token_ms" in gen
print(f"tokens={len(toks)} tok_per_s={gen['tok_per_s']} prefill={gen['prefill_ms']} decode={gen['decode_ms']}")
print("SMOKE OK")
