"""Adapter residency policy: lazy load, one LoRA at a time, swap on change."""
import threading

import pytest

from router.residency import AdapterResidency, Backend


class FakeBackend:
    """Records every heavy operation so the policy can be asserted."""

    def __init__(self):
        self.ops = []
        self.fail_attach = set()

    def backend(self):
        return Backend(
            load_base=lambda: self._op("load_base", "BASE"),
            unload_base=lambda b: self._op("unload_base"),
            attach_lora=self._attach,
            detach_lora=lambda m: self._op("detach:" + m.split(":")[1], "BASE"),
            load_deepfake=lambda: self._op("load_deepfake", ("DF", None)),
            unload_deepfake=lambda d: self._op("unload_deepfake"),
        )

    def _op(self, name, ret=None):
        self.ops.append(name)
        return ret

    def _attach(self, base, task):
        assert base == "BASE", "LoRA must attach to the bare base"
        if task in self.fail_attach:
            raise RuntimeError("missing adapter keys")
        return self._op("attach:" + task, "PEFT:" + task)


def make(policy="swap"):
    fb = FakeBackend()
    return fb, AdapterResidency(fb.backend(), policy=policy)


def test_nothing_loads_until_needed():
    fb, r = make()
    assert fb.ops == []
    assert r.status()["base_loaded"] is False


def test_lora_swaps_on_task_change_and_is_reused_when_unchanged():
    fb, r = make()
    a = r.activate("reasoning")
    assert a.loaded == ["base", "reasoning"]
    b = r.activate("reasoning")
    assert b.loaded == [] and "reasoning" in b.reused
    c = r.activate("anomaly")
    assert c.unloaded == ["reasoning"] and c.loaded == ["anomaly"]
    assert r.lora == "anomaly"
    assert fb.ops == ["load_base", "attach:reasoning", "detach:reasoning",
                      "attach:anomaly"]


def test_general_runs_bare_base():
    fb, r = make()
    r.activate("anomaly")
    g = r.activate("general")
    assert g.unloaded == ["anomaly"]
    assert r.lora is None and r.runner() == "BASE"


def test_swap_policy_keeps_one_backbone_resident():
    fb, r = make("swap")
    r.activate("reasoning")
    d = r.activate("deepfake")
    assert d.unloaded == ["reasoning", "base"] and d.loaded == ["deepfake"]
    assert r.base is None and r.deepfake is not None
    back = r.activate("reasoning")   # the other adapter can load again
    assert back.unloaded == ["deepfake"]
    assert back.loaded == ["base", "reasoning"]
    assert r.deepfake is None


def test_keep_policy_caches_backbones_but_still_one_lora():
    fb, r = make("keep")
    r.activate("reasoning")
    d = r.activate("deepfake")
    assert d.unloaded == ["reasoning"] and r.base == "BASE"
    a = r.activate("anomaly")
    assert a.loaded == ["anomaly"] and "base" in a.reused
    assert r.deepfake is not None


def test_failed_attach_drops_base_and_recovers():
    fb, r = make()
    fb.fail_attach.add("anomaly")
    with pytest.raises(RuntimeError):
        r.activate("anomaly")
    assert r.base is None and r.lora is None
    ok = r.activate("reasoning")
    assert ok.loaded == ["base", "reasoning"]


def test_unknown_task_rejected():
    _, r = make()
    with pytest.raises(ValueError):
        r.activate("unknown_domain")


def test_finance_uses_same_base_and_releases_previous_adapter():
    fb, r = make()
    r.activate('reasoning')
    change = r.activate('finance')
    assert change.unloaded == ['reasoning']
    assert change.loaded == ['finance']
    assert r.base == 'BASE' and r.lora == 'finance'
    bare = r.activate('general')
    assert bare.unloaded == ['finance'] and r.lora is None


def test_lock_can_be_released_from_another_thread():
    # SSE generators resume on different threadpool threads.
    _, r = make()
    r.lock.acquire()
    t = threading.Thread(target=r.lock.release)
    t.start(); t.join()
    assert not r.lock.locked()
