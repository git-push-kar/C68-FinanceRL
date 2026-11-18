import sys, os, json
sys.path.insert(0, r"C:\Users\Admin\Desktop\final")
os.environ.pop("FINAL_STUB_GENERATION", None)
from fastapi.testclient import TestClient
import api.server as srv

c = TestClient(srv.app)
with c.stream("POST", "/chat/stream",
              json={"message": "is this face real or fake",
                    "image_base64": "AAAA", "max_new_tokens": 10}) as s:
    print("HTTP", s.status_code, flush=True)
    for line in s.iter_lines():
        if line and line.startswith("data:"):
            print(line[:300], flush=True)
print("END", flush=True)
