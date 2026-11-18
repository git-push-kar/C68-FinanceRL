# Master Context Transfer & Integration Record: Unified Polyvalent + Finance RL
**Date:** 19 November 2026 · **Backbone:** InternVL3-2B · **Router:** LAYA System-1 Gate · **Status:** Fully Integrated, Conclusive, 75/75 Tests Passing

---

## 1. Executive Summary

This document records the complete integration of the **Finance RL specialist** into the unified **Polyvalent** multi-specialist platform located at `c:\Users\Admin\Desktop\final`. 

The Polyvalent demo now hosts four fully coordinated operational specialists sharing a single pristine **InternVL3-2B** base model under the **LAYA** router and dynamic **AdapterResidency** swapping architecture:
1. **General Chat Specialist:** Pure base model (`InternVL3-2B`) for broad conversational knowledge, reasoning, and programming.
2. **Logical Reasoning Specialist:** LoRA ($r=64$) on the shared base for multi-step deductions, mathematical logic, and decision dilemmas.
3. **Anomaly Detection Specialist:** LoRA ($r=16$) on the shared base for Tennessee Eastman Process (TEP) industrial fault isolation, root-cause analysis (RCA), and incident telemetry reasoning.
4. **Finance RL Portfolio Specialist:** State-aware LoRA (`adapters/internvl3_state_rl`, fallback `adapters/finance_policy`) on the shared base for dynamic portfolio optimization across all 49 Nifty constituent equities.

---

## 2. Finance RL Menu & Three Simulation Modes Architecture

The Finance workspace has been elevated to a production-grade paper trading environment featuring **three distinct simulation modes**:

### Mode 1: Final Benchmark Replay (Jan 2025 – Jan 2026)
- **Sessions:** 270 historical trading sessions.
- **Performance:** **+14.53% net return, 1.121 Sharpe ratio, 1.714 Sortino ratio, 8.87% max drawdown, ₹785.24 cumulative fees** (starting capital ₹100,000). Decisively outperforms passive Buy-and-Hold (+12.56%) and untrained baseline (+13.73%).
- **Execution:** Automated daily playback (`▶ Start simulation`) or stepped execution (`Next day`, `Next rebalance` every 5 trading sessions).

### Mode 2: 2026 YTD History Replay (Jan 1, 2026 – Yesterday)
- **Sessions:** 216+ continuous trading sessions from January 1, 2026 through the most recent historical close.
- **Engine:** Managed by `YTDReplaySession` in `finance/live_manager.py`.
- **Purpose:** Replays realistic 2026 daily trading dynamics with mark-to-market valuations and portfolio state accumulation, preparing the portfolio state for live continuation.

### Mode 3: Live Real-Time Market Trading (Today's Live Session)
- **Engine:** Managed by `LiveRealtimeSession` and `LiveMarketService` in `finance/live_manager.py`.
- **Features:**
  1. **`🔄 Fetch Live Quotes` (`POST /finance/paper/live/refresh`):** Fetches real-time intraday quotes for all 49 Nifty equities via Yahoo Finance, falling back seamlessly to cached market dynamics (`data/finance_paper/live_market_cache.npz`) if offline.
  2. **`🤖 Run Model Decision` (`POST /finance/paper/live/decide`):** Feeds live market prices, technical momentum indicators, and current portfolio holdings into the state-aware InternVL3-2B + LoRA RL adapter to compute optimal target allocations.
  3. **`📌 Settle & Close Today` (`POST /finance/paper/live/settle`):** Commits the current trading session, calculates realized gains/losses, updates cash and equity, logs transaction records, and saves the portfolio state directly to disk.
  4. **`⚡ Step Live Day` (`POST /finance/paper/sessions/live_market_session/advance`):** Single-click atomic execution of market data update, model decision inference, whole-share lot execution, and day advance.

---

## 3. Persistent JSON Storage Architecture

The live portfolio state is persisted across server reboots without loss of accounting history:
- **Primary Store:** `data/finance_paper/live_portfolio_store.json`
- **Automatic Backup:** `data/finance_paper/live_portfolio_store.backup.json`

### Storage Schema:
```json
{
  "mode": "live",
  "settled_date": "2026-11-17",
  "settled_equity": 99793.39,
  "capital": 100000.0,
  "cash": 2.46e-15,
  "fees": 598.68,
  "holdings": [
    {"symbol": "RELIANCE", "shares": 12, "last_price": 2450.0, "value": 29400.0, "weight": 0.2946, "cost_basis": 2400.0, "pnl": 600.0},
    ...
  ],
  "history": [
    {"date": "2026-01-02", "equity": 100000.0, "cash": 100000.0, "fees": 0.0},
    ...
  ],
  "sessions_count": 216,
  "updated_at": "2026-11-18T17:20:13.506996"
}
```

---

## 4. One-Command Execution

The unified demo can be launched with a single click or command:

### Windows Batch Launcher: `run_demo.bat`
```bat
@echo off
set "PY_EXE=C:\Users\Admin\AppData\Local\Programs\Python\Python312\python.exe"
...
start "" "http://127.0.0.1:8000"
%PY_EXE% -m uvicorn api.server:app --host 127.0.0.1 --port 8000 --workers 1
```

### PowerShell Launcher: `Start-Demo.ps1`
```powershell
.\Start-Demo.ps1
```

Both launchers auto-detect Python 3.12, start the unified FastAPI/Uvicorn backend on port 8000, and immediately open the browser interface at `http://127.0.0.1:8000`.

---

## 5. Verification & Regression Test Suite

All **75/75 tests pass** across the unified test suite (`pytest -q tests`):
- `tests/test_three_simulation_modes.py`: Validates Mode 1 (Benchmark Replay), Mode 2 (YTD History Replay), and Mode 3 (Live Real-Time Market trading, live refresh, model decision, and JSON settlement).
- `tests/test_internvl3_state_rl_simulation.py`: Validates RL state policy and whole-share cash accounting.
- `tests/test_live_market_simulation.py`: Validates live sessions and daily step progression.
- `tests/test_residency.py`: Validates adapter residency lease, swap policy, and eviction locks.
- `tests/test_api_stub.py`: Validates end-to-end chat routing, deepfake preservation, and anomaly reporting.
- `tests/test_anomaly_classification.py`: Validates TEP anomaly detector and hybrid classifier.
- `tests/test_multiface_pipeline.py`: Validates YOLO multi-face detection and bounding-box reasoning.
- `tests/test_router_cases.py`: Validates LAYA System-1 routing accuracy across multimodal inputs.
- `tests/test_paper.py` & `test_finance_verification.py`: Validates paper trading accounting invariants.
