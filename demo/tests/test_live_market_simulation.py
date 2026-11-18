"""Tests for Live Real-Time Market Simulation Mode and Persistent Storage."""

import pytest
from fastapi.testclient import TestClient
from pathlib import Path
import sys

DEMO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DEMO_ROOT))

from api.server import app
from finance.live_manager import get_live_manager, STORE_PATH


def test_live_manager_snapshot():
    """Verify LivePortfolioManager loads persistent state and produces a valid snapshot."""
    mgr = get_live_manager()
    snap = mgr.snapshot()

    assert snap['mode'] == 'live'
    assert snap['is_live'] is True
    assert snap['date'] >= '2026-11-17'
    assert snap['initial_date'] <= '2026-01-02'
    assert snap['capital'] == 100000.0
    assert snap['equity'] > 0
    assert len(snap['holdings']) == 49
    assert len(snap['history']) > 200
    assert len(snap['activities']) >= 40
    assert STORE_PATH.exists()


def test_live_session_api_endpoints():
    """Verify API endpoints for live session creation, snapshot, and refresh."""
    client = TestClient(app)

    # 1. Start live session
    res = client.post('/finance/paper/sessions', json={'capital': 100000.0, 'mode': 'live'})
    assert res.status_code == 200
    data = res.json()
    assert data['session_id'] == 'live_market_session'
    assert data['is_live'] is True
    assert len(data['holdings']) == 49

    # 2. Get live snapshot
    res_snap = client.get('/finance/paper/sessions/live_market_session')
    assert res_snap.status_code == 200
    snap = res_snap.json()
    assert snap['session_id'] == 'live_market_session'
    assert snap['equity'] == data['equity']

    # 3. Live refresh endpoint
    res_ref = client.post('/finance/paper/live/refresh')
    assert res_ref.status_code == 200
    ref_data = res_ref.json()
    assert ref_data['session_id'] == 'live_market_session'
    assert ref_data['is_live'] is True
