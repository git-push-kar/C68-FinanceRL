# Adapters and base model (in-repo)

The base model and every adapter are **copied into this repository** so the
service runs from here. The training runs keep their originals; the paths
below are the copies.

Base model (shared by every specialist):

`models/InternVL3-2B` — copied from
`C:\Users\Admin\Desktop\LY_C68\deepfake_project\models\InternVL3-2B`

| Routed task | Path | Format | Copied from |
|---|---|---|---|
| `anomaly` | `adapters/anomaly` | PEFT LoRA r=16 / a=32 on q,k,v,o,gate,up,down | `C:\Users\Admin\Desktop\anomaly\outputs\tep_rca_internvl3_2b\internvl3_2b_pristine_rca` |
| `reasoning` | `adapters/reasoning` | PEFT LoRA r=64 / a=128 on q,k,v,o,gate,up,down | `C:\Users\Admin\Desktop\C68-LR\models_v3\adapter\adapter` |
| `deepfake` | `adapters/deepfake` | `llm_adapter/` (r=64), `vision_adapter/` (r=16), `custom_heads/custom_heads.safetensors` | `C:\Users\Admin\Desktop\LY_C68\deepfake_project\runs\intern_v2_ground\adapters` |
| `general` | - | pristine base, no adapter | - |

Adapters load lazily, on the first request Laya routes to their task
(`router/residency.py`). At most one of `anomaly` / `reasoning` is attached
to the shared base at a time. A task change detaches the old LoRA (unwrapped,
never merged) and attaches the new one. `general` runs the bare base.
`deepfake` runs in the dedicated `DeepfakeReasoningModel` wrapper (its
classifier heads have no InternVL counterpart). Under the default
`FINAL_ADAPTER_POLICY=swap`, the wrapper and the shared base are never
resident together.

`api/server.py` resolves these paths from the repository root, so no
environment variables are needed. Override any path with `FINAL_BASE_DIR`,
`FINAL_ANOMALY_DIR`, `FINAL_REASONING_DIR`, `FINAL_DEEPFAKE_LLM_DIR`,
`FINAL_DEEPFAKE_VISION_DIR`, `FINAL_CUSTOM_HEADS`.

Weights are git-ignored (`.gitignore`: `models/`, `*.safetensors`); the small
`adapter_config.json` files stay tracked.
