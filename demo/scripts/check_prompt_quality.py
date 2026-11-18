import sys, os, time
sys.path.insert(0, r"C:\Users\Admin\Desktop\final")
os.environ.pop("FINAL_STUB_GENERATION", None)
import router._torch_shim  # noqa: F401
from router import AdapterService

from api.server import get_service
svc = get_service()

tests = [
    ("general", [], "hi", 30),
    ("anomaly", [], "XMEAS_7 reactor pressure high, XMV_3 valve stuck. Diagnose the fault.", 120),
]
for task, hist, msg, ntok in tests:
    print(f"===== {task}: {msg[:50]} =====", flush=True)
    print(svc.select(task), flush=True)
    prompt = svc.format_prompt(task, hist, msg)
    print("PROMPT HEAD:", repr(prompt[:200]), flush=True)
    t0 = time.perf_counter()
    out = []
    for chunk in svc.stream(prompt, max_new_tokens=ntok, stub=False,
                            repetition_penalty=1.1):
        if chunk.get("event") == "token":
            out.append(chunk.get("text", ""))
        elif chunk.get("event") == "notice":
            out.append(f"[NOTICE {chunk.get('text','')[:150]}]")
        elif chunk.get("event") == "generation_receipt":
            print("TOK/S:", chunk["tok_per_s"], "TOKENS:", chunk["tokens"], flush=True)
    print("ANSWER:", "".join(out)[:800], flush=True)
    print(f"({time.perf_counter()-t0:.1f}s total)", flush=True)
print("PROMPT-CHECK DONE", flush=True)
