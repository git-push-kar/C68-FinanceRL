# Nifty paper-trading demonstration

**State-Aware Portfolio RL Implementation (Merged November 2026):** The paper trading simulation engine now directly integrates the state-aware `InternVL3-2B + LoRA RL` adapter (`adapters/internvl3_state_rl`). Unlike the earlier static strategy mixture, this policy directly ingests and conditions upon live portfolio state (cash, equity, active weights, affordable lots) and market state across all 49 Nifty equities (trailing 20-day returns, 60-day returns, 20-day volatility, and unit share prices).

The fine-tuned model evaluates all 49 stocks, computes whole-share integer allocations under 10% maximum position limits and turnover hold-gating (<12%), and generates grounded natural-language investment rationales for its top allocations. On the continuous January 2025–January 2026 primary test set, this model achieves **+14.53% net return, 1.121 Sharpe, and 8.87% max drawdown**, convincingly outperforming both passive Buy-and-Hold (+12.56%, 0.944 Sharpe) and the Untrained LLM baseline (+13.73%, 1.060 Sharpe).

The selected InternVL3-2B Nifty RL adapter (`adapters/internvl3_state_rl`, SHA256: `749d98f2420b5ce601b10d1b191dbb2d9985ecac5ce9f290b55defa588c1d340`) controls the historical paper account. Backward compatibility with the legacy strategy-mixture reference is preserved for regression testing.

## Simulation Modes in the Interface

The paper trading interface now features a **Simulation Mode** dropdown selector with two operating modes:

1. **Preset Period (Jan 2025 – Jan 2026):**
   - Replays the 270-session primary retrospective test set.
   - Interactive day-by-day autonomous playback, step-by-step advance, and rebalance inspection.
   - Preserves benchmark comparisons against Untrained LLM and Buy & Hold.

2. **Live Real-Time Market (Jan 2026 – Present):**
   - **Forward Real-Time Mode:** Tracks the portfolio actively managed by the model starting from January 1, 2026 up to today's market session (`2026-11-18`).
   - **Live Market Fetching:** Uses `yfinance` to fetch live daily quotes across all 49 Nifty constituent equities from the National Stock Exchange of India.
   - **Continuous Day-to-Day Resumption:** Persists portfolio holdings, cash, cost basis, cumulative P&L, and model rationales in `data/finance_paper/live_portfolio_store.json`.
   - **Next Trading Day Progression:** When tomorrow (e.g., Nov 19, 2026) arrives, clicking **"⚡ Step Live Day"** or **"🔄 Fetch Live Quotes"** ingests the new day's market data, allows the `InternVL3-2B + LoRA RL` adapter to evaluate the updated portfolio state, executes whole-share trades if rebalance is due, and marks the account to market with the market close. On the day after tomorrow (Nov 20, 2026), it resumes seamlessly from Nov 19's closing state.

## On this PC

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\Start-FinanceDemo.ps1
```

Open http://127.0.0.1:8765/#finance. Start simulation for autonomous playback, or use Next day / Next rebalance. Select View decision for exact transactions and strategy mix. Holdings/activity export the revealed records. Reset portfolio restarts from cash with the same history. Sessions are in memory; restart the server to discard them.

The full test spans 1 January 2025–30 January 2026. No broker connection, today's prices or real money are involved. On scheduled rebalance sessions the actual selected LoRA performs inference over the 49-asset numerical state. If actual inference fails or differs from the selected artifact, the simulation stops rather than substituting recorded action probabilities.

## Recover on another PC

From the GitHub repository root:

```powershell
git lfs install
git lfs pull --include="demo/adapters/internvl3_state_rl/**,demo/adapters/finance_policy/**,demo/data/finance_paper/**"
cd demo
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
.\.venv\Scripts\python.exe -m pip install -r requirements-paper-demo.txt
```

The tested environment is Windows/Python3.12, RTX A5000 / RTX 4060Ti. The launcher never installs dependencies or silently downloads a different model. Install the **full pinned original base** separately:

```powershell
.\.venv\Scripts\python.exe -c "from huggingface_hub import snapshot_download; snapshot_download('OpenGVLab/InternVL3-2B', revision='899155015275a9b7338c7f4677e19c784e0e5a21', local_dir='../models/InternVL3-2B')"
powershell -NoProfile -ExecutionPolicy Bypass -File .\Start-FinanceDemo.ps1 -BaseModelPath ..\models\InternVL3-2B
```

Download requires internet once. Inference then uses offline flags. The base's `model.safetensors` SHA256 is `18b4cdbc8eab790f447feac717f4d0cecbb85caf8df198b3b2401c22506858e7`. Include all snapshot configuration, tokenizer and remote-code files, not only its weight tensor.

## Verification

```powershell
python -m pytest tests -q
```

The 37-test suite checks account/fee parity, causal target construction, references, failure behavior, state-aware 49-asset rebalancing, rationale generation, and residency/transport.

## Research interpretation

Training: July 2, 2018–December 31, 2022; Validation: 2023–2024; Retrospective continuous test: 2025–January 30, 2026.
- Buy & Hold: 12.56% return, 0.944 Sharpe, 10.38% max drawdown.
- Untrained LLM: 13.73% return, 1.060 Sharpe, 8.51% max drawdown.
- Trained InternVL3-2B + LoRA RL: 14.53% return, 1.121 Sharpe, 8.87% max drawdown.

Full method/failure/context documentation is in the repository's `docs/FINANCE_RL_RESEARCH_JOURNAL.md` and `docs/FINANCE_EXPERIMENT_HANDOVER_20261118.md`.
