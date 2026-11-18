"""Verify lazy adapter loading and LR activation (loads the real weights).

Policy under test:
  * `reasoning` first  -> only the LR adapter is resident
  * `general`          -> LR stays resident but inactive
  * `anomaly`          -> anomaly adapter is added lazily
  * switching back     -> no duplicate loads, correct active adapter

No generation happens here, so this only pays the model-load cost.

Run with the uvicorn server STOPPED:
    & $py scripts\\verify_adapter_loading.py
"""
import os
import sys

sys.path.insert(0, r"C:\Users\Admin\Desktop\final")
os.environ.pop("FINAL_STUB_GENERATION", None)

from api.server import get_service
from router.prompts import (GENERAL_SYSTEM, REASONING_SYSTEM,
                            build_messages)

svc = get_service()
ok = True


def prompt_step(task: str, want: str) -> None:
    global ok
    got = build_messages(task, [], "q")[0]["content"]
    good = got == want
    ok &= good
    print(("PASS" if good else "FAIL"), task,
          "system prompt is", want.split(",")[0][:40] + "...", flush=True)


prompt_step("reasoning", REASONING_SYSTEM)
prompt_step("general", GENERAL_SYSTEM)


def step(task, want_loaded, want_active):
    global ok
    sel = svc.select(task)
    got_loaded = svc.loaded_adapters
    got_active = svc.active_adapter
    good = got_loaded == want_loaded and got_active == want_active
    ok &= good
    print(("PASS" if good else "FAIL"), task,
          "-> loaded", got_loaded, "active", got_active,
          "| receipt", sel.get("adapters"), flush=True)


step("reasoning", ["reasoning"], "reasoning")
step("general", ["reasoning"], None)
step("anomaly", ["anomaly", "reasoning"], "anomaly")
step("reasoning", ["anomaly", "reasoning"], "reasoning")
step("general", ["anomaly", "reasoning"], None)
print("ADAPTER LOADING OK" if ok else "ADAPTER LOADING FAILED", flush=True)
