import sys, os, time
sys.path.insert(0, r"C:\Users\Admin\Desktop\final")
os.environ.pop("FINAL_STUB_GENERATION", None)
import router._torch_shim  # noqa: F401
from router import AdapterService

from api.server import get_service
svc = get_service()
print(svc.select("deepfake"), flush=True)
img_path = r"C:\Users\Admin\Desktop\LY_C68\deepfake_project\test_images\1.jpg"
raw = open(img_path, "rb").read()
print("bytes:", len(raw), flush=True)
n = 0
t0 = time.perf_counter()
for chunk in svc.analyze_image_stream(raw, max_new_tokens=120, stub=False):
    ev = chunk.get("event")
    if ev == "verdict":
        print("VERDICT:", {k: chunk[k] for k in ("verdict", "confidence", "hotspots", "telemetry_ms")}, flush=True)
        print("SPECTRAL:", str(chunk.get("spectral"))[:120], flush=True)
    elif ev == "token":
        n += 1
        if n <= 12:
            print("TOK:", repr(chunk.get("text", "")), flush=True)
    elif ev == "notice":
        print("NOTICE:", str(chunk.get("text"))[:200], flush=True)
    elif ev == "generation_receipt":
        print("TOK/S:", chunk["tok_per_s"], "TOKENS:", chunk["tokens"], flush=True)
print(f"TOTAL {time.perf_counter()-t0:.1f}s TOKENS={n}", flush=True)
print("DEEPFAKE-LIVE DONE", flush=True)
