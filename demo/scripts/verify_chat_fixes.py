import sys, os, json
sys.path.insert(0, r"C:\Users\Admin\Desktop\final")
os.environ.pop("FINAL_STUB_GENERATION", None)
import router._torch_shim  # noqa: F401
from fastapi.testclient import TestClient
import api.server as srv

c = TestClient(srv.app)

print("--- hi (general, real) ---", flush=True)
with c.stream("POST", "/chat/stream",
              json={"message": "hi", "max_new_tokens": 40}) as s:
    for line in s.iter_lines():
        if line and line.startswith("data:"):
            ev = json.loads(line[6:])
            print(ev["event"], repr(ev.get("text", "")[:160]), flush=True)

print("--- deepfake text-only (expect notice) ---", flush=True)
with c.stream("POST", "/chat/stream",
              json={"message": "state if this image is real or fake",
                    "max_new_tokens": 40}) as s:
    for line in s.iter_lines():
        if line and line.startswith("data:"):
            ev = json.loads(line[6:])
            print(ev["event"], repr(ev.get("text", "")[:160]), flush=True)
print("CHAT-FIX DONE", flush=True)
