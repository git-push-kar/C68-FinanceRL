import sys, os, time
sys.path.insert(0, r"C:\Users\Admin\Desktop\final")
os.environ.pop("FINAL_STUB_GENERATION", None)
import router._torch_shim  # noqa: F401
from router import AdapterService

from api.server import get_service
svc = get_service()
t0 = time.perf_counter()
sel = svc.select("anomaly")
print("SELECT:", sel, f"{time.perf_counter()-t0:.1f}s")
n = 0
t0 = time.perf_counter()
for chunk in svc.stream("XMEAS_7 reactor pressure high. Diagnose the fault.",
                        max_new_tokens=20, stub=False):
    if chunk.get("event") == "token":
        n += 1
        print(f"[{n}] {chunk['text']!r} +{chunk['latency_ms']}ms")
    elif chunk.get("event") == "generation_receipt":
        print("RECEIPT:", chunk)
print(f"TOTAL {time.perf_counter()-t0:.1f}s")
