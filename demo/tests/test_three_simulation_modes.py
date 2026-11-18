import pytest
import datetime
from pathlib import Path
from fastapi.testclient import TestClient

DEMO_ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(DEMO_ROOT))

from api.server import app
from finance.live_manager import (
    LiveMarketService,
    YTDReplaySession,
    LiveRealtimeSession,
    get_ytd_session,
    get_live_realtime_session
)


@pytest.fixture
def client():
    return TestClient(app)


def test_mode1_benchmark_replay(client):
    """Verify Mode 1: Final Benchmark Replay (Jan 2025 - Jan 2026)."""
    res = client.post("/finance/paper/sessions", json={"capital": 100000.0, "mode": "benchmark"})
    assert res.status_code == 200, res.text
    data = res.json()
    assert data["capital"] == 100000.0
    assert data["total_sessions"] == 270
    assert "session_id" in data
    assert data["session_id"] != "live_market_session"
    assert data["session_id"] != "ytd_history_session"


def test_mode2_ytd_history_replay(client):
    """Verify Mode 2: 2026 YTD History Replay (Jan 1, 2026 - Yesterday)."""
    res = client.post("/finance/paper/sessions", json={"capital": 100000.0, "mode": "ytd"})
    assert res.status_code == 200, res.text
    data = res.json()
    assert data["session_id"] == "ytd_history_session"
    assert data["mode"] == "ytd"
    assert data["total_sessions"] >= 200
    assert data["initial_date"] == "2026-01-02"

    # Step day forward
    res_adv = client.post("/finance/paper/sessions/ytd_history_session/advance", json={"mode": "day", "expected_sessions": data["sessions"]})
    assert res_adv.status_code == 200
    data_adv = res_adv.json()
    assert data_adv["sessions"] == data["sessions"] + 1

    # Step rebalance forward
    res_reb = client.post("/finance/paper/sessions/ytd_history_session/advance", json={"mode": "rebalance", "expected_sessions": data_adv["sessions"]})
    assert res_reb.status_code == 200
    data_reb = res_reb.json()
    assert data_reb["sessions"] >= data_adv["sessions"]


def test_mode3_live_realtime_trading(client):
    """Verify Mode 3: Live Real-Time Market Trading & Day Settlement."""
    res = client.post("/finance/paper/sessions", json={"capital": 100000.0, "mode": "live"})
    assert res.status_code == 200, res.text
    data = res.json()
    assert data["session_id"] == "live_market_session"
    assert data["mode"] == "live"
    assert data["is_live"] is True
    assert "settled_date" in data
    assert data["equity"] > 0
    assert len(data["holdings"]) == 49

    # Test live refresh endpoint
    res_ref = client.post("/finance/paper/live/refresh")
    assert res_ref.status_code == 200
    data_ref = res_ref.json()
    assert data_ref["session_id"] == "live_market_session"
    assert "market_status" in data_ref

    # Test live decision endpoint
    res_dec = client.post("/finance/paper/live/decide")
    assert res_dec.status_code == 200
    data_dec = res_dec.json()
    assert data_dec["equity"] > 0

    # Test day settlement into permanent storage (forced date to avoid altering baseline date)
    dummy_date = "2026-11-30"
    res_set = client.post("/finance/paper/live/settle", json={"target_date": dummy_date})
    assert res_set.status_code == 200
    data_set = res_set.json()
    assert data_set["settled_date"] == dummy_date
    assert data_set["is_settled_today"] is True
