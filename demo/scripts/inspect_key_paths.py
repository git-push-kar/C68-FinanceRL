import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))
import router._torch_shim  # noqa: F401
from transformers import AutoModel

base = AutoModel.from_pretrained(
    str(_REPO / "models" / "InternVL3-2B"),
    trust_remote_code=True, torch_dtype="auto", device_map="cpu")
names = [n for n, _ in base.named_modules()
         if "layers.0.self_attn.q_proj" in n or "layers.0.mlp.down_proj" in n]
print("LIVE LLM PATHS:", names[:4])
vnames = [n for n, _ in base.named_modules() if "qkv" in n][:3]
print("LIVE VISION PATHS:", vnames)

from safetensors import safe_open
for label, p in [
    ("deepfake_llm",
     _REPO / "adapters" / "deepfake" / "llm_adapter" / "adapter_model.safetensors"),
    ("anomaly", _REPO / "adapters" / "anomaly" / "adapter_model.safetensors"),
]:
    keys = list(safe_open(str(p), framework="pt").keys())
    print(label, "n=", len(keys), "sample:", keys[0], "|", keys[1])
