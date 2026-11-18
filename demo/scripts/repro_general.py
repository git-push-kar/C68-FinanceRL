import sys, os, logging
sys.path.insert(0, r"C:\Users\Admin\Desktop\final")
os.environ.pop("FINAL_STUB_GENERATION", None)
logging.basicConfig(level=logging.WARNING)
import router._torch_shim  # noqa: F401
from router import AdapterService

from api.server import get_service
svc = get_service()
print(svc.select("general"), flush=True)
n = 0
for chunk in svc.stream("Say hello briefly.", max_new_tokens=6, stub=False):
    print(chunk.get("event"), repr(chunk.get("text", "")[:60]), flush=True)
    n += 1
print("CHUNKS:", n, flush=True)
