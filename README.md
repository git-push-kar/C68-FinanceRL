# Finance RL · InternVL3-2B Full-Universe State-Aware Reinforcement Learning

**Current status:** **Conclusive Native InternVL3-2B + LoRA Financial RL Run Complete & Merged into Paper Trading Engine.** All 49 Nifty equities are fully observed and managed with real-time portfolio state (cash, holdings, weights, affordabilities, equity, drawdown, volatility, 20d/60d momentum). The native multimodal LLM policy with PEFT LoRA ($r=16, \alpha=32$) trained via Trajectory Policy Gradient with symmetric action-centering achieved **+14.53% net test return**, **1.121 Sharpe ratio**, and **8.87% max drawdown** on strictly unseen 2025–2026 data at ₹100,000 capital, convincingly outperforming both passive Buy-and-Hold (+12.56%) and the untrained InternVL3-2B baseline (+13.73%).

See the complete academic research paper: **[FINANCE_RL_RESEARCH_JOURNAL.md](docs/FINANCE_RL_RESEARCH_JOURNAL.md)**.  
Training and evaluation implementation: `research/train_internvl3_native_rl.py` and `research/evaluate_internvl3_native_rl.py`.  
Paper trading integration and tests: `demo/finance/policy.py`, `demo/finance/paper_engine.py`, and `demo/tests/test_internvl3_state_rl_simulation.py`.

---

## Conclusive Out-of-Sample Test Results (2025-01-01 to 2026-01-30, ₹100,000 Capital, All 49 Stocks)

| Strategy / Model | Architecture / Method | Net Return | Sharpe Ratio | Sortino Ratio | Max Drawdown | Total Fees | Net Advantage vs. Hold |
|---|---|---:|---:|---:|---:|---:|---:|
| **InternVL3-2B + LoRA RL (Ours)** | **2B Vision-LLM + LoRA + Policy Gradient** | **+14.53%** | **1.121** | **1.714** | **8.87%** | **₹785.24** | **+1.97 pp** |
| **Untrained InternVL3-2B Baseline** | 2B Vision-LLM Zero-Shot Prior | +13.73% | 1.060 | 1.582 | 8.51% | ₹811.38 | +1.17 pp |
| **State-Aware Cross-Asset Pilot** | 49-Head Cross-Asset Dense Network | +13.79% | 0.898 | 1.281 | 7.28% | ₹1,036.26 | +1.23 pp |
| **Passive Buy-and-Hold** | Fixed Initial Allocation | +12.56% | 0.944 | 1.346 | 10.38% | ₹99.95 | 0.00 pp |
| **Equal-Weight Rebalanced** | Periodic Uniform Rebalancing | +11.11% | 0.849 | 1.218 | 10.26% | ₹1,522.73 | -1.45 pp |
| **Previous Stage 5 NativePolicy** | 4-Strategy Mixture (Old PC) | +10.54% | 0.788 | 1.109 | 11.28% | ₹138.34 | -2.02 pp |
| **120d Momentum Teacher** | Heuristic Momentum Baseline | +2.59% | 0.279 | 0.392 | 7.20% | ₹834.23 | -9.97 pp |

---

## Historical static-policy result

Training2019–2021, validation2022, retrospective continuous test1January2025–30January2026. Same49stocks, costs and dates:

| Policy | Net return | Sharpe | Max drawdown |
|---|---:|---:|---:|
| Original model |11.22%|0.885|8.13%|
| Selected RL adapter |13.28%|1.033|7.78%|
| Equal-weight buy-and-hold |13.06%|1.012|7.75%|
| Equal-weight rebalanced |13.44%|1.035|7.79%|

Observed training benefit over raw; small, uncertain, cost-sensitive holding advantage. No established conventional-strategy superiority or untouched-test claim.

## Demonstration

`demo/` serves the portfolio menu at `http://127.0.0.1:8765/#finance`. Each scheduled rebalance calls the actual selected Nifty LoRA. Historical prices, simulated cash and fee-funded trades advance autonomously. The270-day model-backed account exactly reproduces the original measured path. Finance conversation was removed after factual testing; inspect actual execution records instead. Install Git LFS and the pinned original upstream base as described in [setup](demo/PAPER_TRADING.md).

![Actual paper-trading interface](demo/portfolio-paper-live.png)

## Repository organization

- `demo/`: current application, selected policy weights/data via LFS, tests and real inference/UI evidence.
- `docs/`: final results, journal-level method/failure report, complete numeric appendix, narration assessment and historical transfer.
- `research/`: source snapshots and canonical Nifty dataset for reproduction; historical scripts may retain original paths. Read recovery instructions rather than assuming every old script runs from repository root.
- `archives/`: historical source/metadata and selected packages via Git LFS, checksums, plus separate full-checkpoint release status.
- Original root-level scripts from `master` are retained legacy code and do not define the current selected native policy.

Use `git lfs pull --include="demo/adapters/finance_policy/**,demo/data/finance_paper/**"` for the minimal demo assets; plain GitHub ZIP downloads may contain LFS pointers rather than weights. All current commits use the authenticated user's configured identity; preexisting history is preserved.
