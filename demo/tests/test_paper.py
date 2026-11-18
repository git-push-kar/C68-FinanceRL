"""Accounting/causality checks against independently saved research outputs."""
import json
from pathlib import Path
import numpy as np
import pytest
from finance.paper_engine import Panel, PaperAccount

ROOT = Path(__file__).resolve().parents[1]


def reference():
    p = Panel()
    q = np.asarray(p.manifest['recorded_probabilities']['seed42_64'], dtype=float)
    return p, lambda: (q, {'mode': 'unit_test_reference'})


def test_entire_path_matches_original_research():
    p, predict = reference()
    a = PaperAccount(p)
    while a.step(predict):
        pass
    expected = json.loads((ROOT/'finance/evaluation/paper_golden.json').read_text())
    np.testing.assert_allclose([h['equity'] for h in a.history], expected['trained_equity'], rtol=0, atol=1e-7)
    np.testing.assert_allclose([h['untrained'] for h in a.history], expected['untrained_equity'], rtol=0, atol=1e-7)
    np.testing.assert_allclose([h['buy_hold'] for h in a.history], expected['buy_hold_equity'], rtol=0, atol=1e-7)
    assert a.fees == pytest.approx(expected['trained_fees'], abs=1e-7)
    assert len(a.activities) == 54


def test_model_failure_leaves_account_untraded():
    p, _ = reference()
    a = PaperAccount(p)
    def fail():
        raise RuntimeError('Model unavailable')
    with pytest.raises(RuntimeError):
        a.step(fail)
    assert a.n == 0 and a.equity == 100000 and a.fees == 0 and not a.activities


def test_stub_service_cannot_make_a_paper_decision():
    from types import SimpleNamespace
    from finance.policy import predict
    with pytest.raises(RuntimeError, match='stubs are disabled'):
        predict(SimpleNamespace(_stub_backend=True))


def test_only_scheduled_days_invoke_model_and_signals_are_lagged():
    p, predict = reference()
    a = PaperAccount(p)
    calls = []
    def counted():
        calls.append(a.n)
        return predict()
    for _ in range(11):
        a.step(counted)
    assert calls == [0, 5, 10]
    assert all(x['signal_date'] < x['execution_date'] for x in a.activities)
    assert len(a.snapshot()['history']) == 11
    assert a.snapshot()['date'] == a.history[-1]['date']


def test_future_prices_cannot_change_allocation():
    p, _ = reference()
    t = p.targets(p.first)
    changed = Panel()
    changed.prices[p.first:] *= np.linspace(.6, 1.7, len(changed.prices)-p.first)[:, None]
    np.testing.assert_array_equal(t, changed.targets(changed.first))


def test_capital_scaling_preserves_policy_and_returns():
    p, predict = reference()
    a, b = PaperAccount(p), PaperAccount(p, 250000)
    for _ in range(12):
        a.step(predict)
        b.step(predict)
    assert b.equity == pytest.approx(a.equity*2.5, abs=1e-7)
    assert b.fees == pytest.approx(a.fees*2.5, abs=1e-7)


def test_duplicate_advance_is_rejected(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient
    from api.server import app
    import api.paper as api
    import api.server as server
    _, predict = reference()
    monkeypatch.setattr(api, 'ASSESSMENT_MARKER', tmp_path/'absent.json')
    monkeypatch.setattr(api, 'predict', lambda service: predict())
    monkeypatch.setattr(server, 'get_service', lambda: None)
    c = TestClient(app)
    j = c.post('/finance/paper/sessions', json={'capital': 100000}).json()
    key = j['session_id']
    request = {'mode': 'day', 'expected_sessions': 0}
    assert c.post(f'/finance/paper/sessions/{key}/advance', json=request).status_code == 200
    assert c.post(f'/finance/paper/sessions/{key}/advance', json=request).status_code == 409
    assert c.get(f'/finance/paper/sessions/{key}').json()['sessions'] == 1
    c.delete(f'/finance/paper/sessions/{key}')
