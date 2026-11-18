import sys, os, time, traceback
sys.path.insert(0, r"C:\Users\Admin\Desktop\final")
os.environ.pop("FINAL_STUB_GENERATION", None)
import router._torch_shim  # noqa: F401
from router import AdapterService

from api.server import get_service
svc = get_service()

for task, prompt in [("general", "Say hello briefly."),
                     ("anomaly", "XMEAS_7 high. Diagnose."),
                     ("deepfake", "Describe this in one sentence.")]:
    print(f"===== select({task}) =====", flush=True)
    try:
        print(svc.select(task), flush=True)
    except Exception:
        traceback.print_exc()
        continue
    try:
        for chunk in svc.stream(prompt, max_new_tokens=6, stub=False):
            if chunk.get("event") == "token":
                print("TOK:", repr(chunk["text"]), flush=True)
            elif chunk.get("event") == "notice":
                print("NOTICE:", chunk["text"][:200], flush=True)
            elif chunk.get("event") == "generation_receipt":
                print("TOK/S:", chunk["tok_per_s"], flush=True)
    except Exception:
        traceback.print_exc()
print("REPRO DONE", flush=True)
