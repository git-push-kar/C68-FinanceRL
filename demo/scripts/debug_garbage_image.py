import sys, os, json, traceback
sys.path.insert(0, r"C:\Users\Admin\Desktop\final")
from fastapi.testclient import TestClient
import api.server as srv

c = TestClient(srv.app, raise_server_exceptions=False)
with c.stream("POST", "/chat/stream",
              json={"message": "is this face real or fake",
                    "image_base64": "AAAA", "max_new_tokens": 6}) as s:
    print("HTTP", s.status_code, flush=True)
    try:
        for line in s.iter_lines():
            if line and line.startswith("data:"):
                print(line[:220], flush=True)
    except Exception:
        traceback.print_exc()
print("END", flush=True)
