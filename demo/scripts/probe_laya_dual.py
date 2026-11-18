import sys, json
sys.path.insert(0, r"C:\Users\Admin\Desktop\final")
import router._torch_shim  # noqa: F401  (eager torch.compile fallback)
from laya import Router

r = Router()
q = {
    "task": {"type": "choice",
             "instructions": "Route this request to the correct specialist adapter.",
             "criteria": {
                 "deepfake": "face image, real/fake verdict, spectral or forgery artifacts",
                 "anomaly": "TEP sensors XMEAS/XMV, pressure, valve, reactor, fault diagnosis",
                 "general": "general question fitting no specialist"}},
    "is_anomaly": {"type": "noul",
                   "instructions": "The user is asking for TEP sensor fault diagnosis."},
    "is_deepfake": {"type": "noul",
                    "instructions": "The user is asking whether a face image is real or fake."},
}
for text in ["XMEAS_7 reactor pressure high, diagnose the fault",
             "Please tell me if this image is fake or real",
             "hi"]:
    out = r.predict(text, q)
    print("TEXT:", text)
    print(json.dumps(out["answers"], indent=1)[:900])
