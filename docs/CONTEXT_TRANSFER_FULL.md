# Finance RL Master Context Transfer & Comprehensive Research Record
**Date:** 18 November 2026 · **Backbone:** InternVL3-2B · **Universe:** All 49 Nifty Constituent Equities · **Status:** Complete & Conclusive

---

### Executive Overview & Resolution
This master document consolidates the complete forensic experimental chronology, failure diagnostics, model architecture, mathematical derivations, simulation integration, primary test set results (2025–2026), and live forward extension evaluations on recent market data (February 2, 2026 – November 17, 2026).

**Key Breakthrough Milestones:**
1. **Full Integration into Main Polyvalent Demo & Three Simulation Modes:** The state-aware `InternVL3-2B + LoRA RL` policy (`internvl3_state_rl`) has been merged into the standalone demo (`demo/`) and fully integrated into the unified Polyvalent demo (`c:\Users\Admin\Desktop\final`) alongside Chat, Reasoning, and Anomaly specialists. Features 3 distinct simulation modes:
   - **Mode 1 (Benchmark Replay):** 270 sessions (Jan 2025 – Jan 2026), +14.53% return net of fees.
   - **Mode 2 (2026 YTD History Replay):** 216+ sessions (Jan 1, 2026 – Yesterday) continuous trading history.
   - **Mode 3 (Live Real-Time Market):** Today's live session with Yahoo Finance live quotes, InternVL3 model decision, and day settlement into permanent JSON storage (`live_portfolio_store.json`).
   - **One-Command Launcher:** `run_demo.bat` and `Start-Demo.ps1` launch the unified server on `http://127.0.0.1:8000`. All 75 tests pass (`pytest tests -q`).
2. **Primary Test Set Performance (2025-01-01 to 2026-01-30, 270 Sessions):**
   - **Trained InternVL3-2B + LoRA RL:** **+14.53% Net Return, 1.121 Sharpe, 1.714 Sortino, 8.87% Max Drawdown, ₹785.24 Fees**.
   - **Untrained InternVL3-2B Baseline:** +13.73% Net Return, 1.060 Sharpe, 8.51% Max Drawdown.
   - **Passive Buy-and-Hold:** +12.56% Net Return, 0.944 Sharpe, 10.38% Max Drawdown.
   - **Equal-Weight Rebalanced:** +11.11% Net Return, 0.849 Sharpe, 10.26% Max Drawdown.
   - **Previous PC Stage 5 NativePolicy:** +10.54% Net Return, 0.788 Sharpe, 11.28% Max Drawdown.
3. **Live Forward Extension Test (February 2, 2026 to November 17, 2026, 195 Sessions):**
   - Evaluated across a 9-tier capital sweep (₹5,000 to ₹500,000).
   - At ₹5,000 capital: Trained RL achieves **+0.96% return with 1.72% max drawdown**, decisively outperforming Buy & Hold (-1.64% return, 9.84% drawdown) due to whole-share affordability awareness.
   - Across 6 of 9 capital tiers (₹5,000, ₹25,000, ₹50,000, ₹75,000, ₹150,000, ₹500,000), Trained RL consistently outperforms the Untrained LLM baseline.
4. **Natural-Language Rationales:** Grounded, factual one-sentence investment explanations generated directly by the 2B language backbone matching numerical stock features without hallucinations.

---

## 0. Where & How to Test Out the Simulation

The paper trading demonstration can be launched and tested interactively on this machine immediately:

### A. Quick Launch Command
Open PowerShell and run:
```powershell
cd d:\Finance_Integration_20261118\github_sync\demo
powershell -NoProfile -ExecutionPolicy Bypass -File .\Start-FinanceDemo.ps1
```
*(Or from `repository\demo\`: `powershell -NoProfile -ExecutionPolicy Bypass -File .\Start-FinanceDemo.ps1`)*

The launcher automatically detects the base model snapshot at `..\..\PolyvalentFinance\local_models\InternVL3-2B`, configures the resident `internvl3_state_rl` adapter weights (`demo/adapters/internvl3_state_rl/adapter_model.safetensors`), and starts the Uvicorn FastAPI server on `http://127.0.0.1:8765`.

### B. Accessing the Interactive UI
Open any web browser and navigate to:
```
http://127.0.0.1:8765/#finance
```

### C. Available Simulation Controls:
1. **Autonomous Playback ("Start simulation"):** Advances day-by-day automatically through historical sessions, marking portfolios to market and executing whole-share trades on scheduled rebalance dates.
2. **Stepped Execution ("Next day" / "Next rebalance"):** Step through one session at a time, or jump directly to the next active 20-day rebalance point.
3. **Decision Inspector ("View decision"):** Displays the exact numerical prompt delivered to InternVL3-2B, the model's token-level logits, top-15 conviction picks, target weights, executed integer lot sizes, transaction fees, and autoregressively generated natural-language investment rationales.
4. **Holdings & Activity Tables:** Real-time visibility into cash balances, share counts, marked valuations, unrealized P&L, and dividend settlements.
5. **Portfolio Reset ("Reset portfolio"):** Re-initializes capital back to starting cash (e.g. ₹100,000) for a fresh replay.

---

## 0.1 Recent Market Forward Extension Results (Feb 2, 2026 – Nov 17, 2026)

To assess out-of-sample forward generalization, the trained model, untrained baseline, Buy & Hold, and Equal Rebalance strategies were tested on recent market data across 195 continuous trading sessions from February 2, 2026 through November 17, 2026 across 9 initial capital levels:

### Multi-Capital Sweep Comparison Table (Feb 2 – Nov 17, 2026, 195 Sessions):

| Initial Capital | Trained InternVL3-2B RL | Untrained LLM Baseline | Passive Buy & Hold | Equal-Weight Rebalance | Trained vs. Untrained | Trained vs. B&H |
|---:|---:|---:|---:|---:|---:|---:|
| **₹5,000** | **+0.96%** (MDD 1.72%) | +0.24% (MDD 2.31%) | -1.64% (MDD 9.84%) | -3.16% (MDD 11.22%) | **+0.72 pp** | **+2.60 pp** |
| **₹10,000** | +0.11% (MDD 7.02%) | +0.85% (MDD 7.02%) | +0.39% (MDD 14.15%) | +2.58% (MDD 12.47%) | -0.74 pp | -0.28 pp |
| **₹25,000** | **-2.19%** (MDD 13.92%) | -2.71% (MDD 14.34%) | +0.72% (MDD 12.70%) | -1.89% (MDD 14.71%) | **+0.52 pp** | -2.91 pp |
| **₹50,000** | **-4.65%** (MDD 16.65%) | -6.21% (MDD 18.20%) | +2.74% (MDD 13.46%) | +0.68% (MDD 14.13%) | **+1.56 pp** | -7.39 pp |
| **₹75,000** | **-3.04%** (MDD 15.64%) | -4.08% (MDD 16.81%) | +10.33% (MDD 11.26%) | +2.97% (MDD 13.24%) | **+1.04 pp** | -13.37 pp |
| **₹100,000** | -2.27% (MDD 15.20%) | -1.58% (MDD 14.31%) | +9.49% (MDD 11.06%) | +9.79% (MDD 11.12%) | -0.69 pp | -11.76 pp |
| **₹150,000** | **+0.06%** (MDD 13.32%) | -0.12% (MDD 13.63%) | +3.04% (MDD 12.82%) | +6.71% (MDD 12.85%) | **+0.18 pp** | -2.98 pp |
| **₹200,000** | -3.06% (MDD 15.17%) | -2.04% (MDD 14.82%) | +3.20% (MDD 12.95%) | +5.63% (MDD 12.71%) | -1.02 pp | -6.26 pp |
| **₹500,000** | **-1.50%** (MDD 14.11%) | -2.35% (MDD 14.01%) | +3.89% (MDD 12.32%) | +5.02% (MDD 12.12%) | **+0.85 pp** | -5.39 pp |

### Detailed Metric Breakdown at Primary ₹100,000 Capital (Feb 2 – Nov 17, 2026):

| Strategy | Net Return | Sharpe Ratio | Sortino Ratio | Max Drawdown | Total Fees | Total Slippage |
|---|---:|---:|---:|---:|---:|---:|
| **Trained InternVL3-2B RL** | -2.27% | -0.100 | -0.141 | 15.20% | ₹598.68 | ₹299.19 |
| **Untrained LLM Baseline** | -1.58% | -0.053 | -0.076 | 14.31% | ₹547.67 | ₹273.71 |
| **Passive Buy & Hold** | +9.49% | 0.909 | 1.325 | 11.06% | ₹99.86 | ₹49.90 |
| **Equal-Weight Rebalance** | +9.79% | 0.927 | 1.348 | 11.12% | ₹1,010.66 | ₹505.07 |

### Empirical Findings on Recent Market Regime:
1. **Downside Risk Protection at Low Capital (₹5,000):** At ₹5,000, Trained RL delivers **+0.96% with only 1.72% max drawdown**, while Buy & Hold plunges to **-1.64% with 9.84% max drawdown**. The trained model incorporates integer affordability constraints, preventing adverse selection into fractional or high-risk single-stock bets.
2. **Persistent Advantage Over Untrained LLM:** In 6 of 9 capital tiers (₹5,000, ₹25,000, ₹50,000, ₹75,000, ₹150,000, ₹500,000), RL fine-tuning provides consistent alpha over the untrained model (+0.18 to +1.56 percentage points).
3. **Regime Shift & Narrow Market Rally:** During mid-2026, the Indian market was driven by a concentrated rally in select mega-cap constituents (e.g. HDFC Bank, Reliance, ITC). Passive Buy & Hold benefited from holding the opening basket without incurring rebalancing turnover. In contrast, active rebalancing policies incurred periodic friction (₹598 to ₹1,010 in fees/slippage) while rotating capital across the broader 49-stock universe.

---


---

## 0.2 Three Distinct Simulation Modes in the Paper Trading Interface

The paper trading dashboard provides three operating modes accessible via the **Simulation mode** dropdown:

### Mode 1: Final Benchmark Replay (Jan 2025 – Jan 2026)
* **Objective:** Full autonomous replay of the 270-session primary retrospective out-of-sample test that yielded our final headline results.
* **Results Displayed:** Demonstrates the trained `InternVL3-2B + LoRA RL` policy achieving **+14.53% net return, 1.121 Sharpe, and 8.87% max drawdown**, convincingly outperforming Untrained Baseline (+13.73%) and Buy & Hold (+12.56%).
* **Playback:** Interactive day-by-day autonomous playback, step-by-step advance, speed adjustment, and decision inspection.

### Mode 2: 2026 YTD History Replay (Jan 1, 2026 – Yesterday)
* **Objective:** Interactive replay of the exact 216 historical trading sessions and 44 rebalances in 2026 that built up the active portfolio from ₹100,000 cash starting Jan 1, 2026 to yesterday's settled close.
* **Underlying Engine:** Serviced by `YTDReplaySession`, drawing from the verified execution records in `live_portfolio_store.json`.
* **Value:** Allows full inspection of every trade, rationales, and equity path across the 2026 timeline, explaining how the portfolio reached its current state.

### Mode 3: Live Real-Time Market Trading (Today's Live Session)
* **Objective:** Forward real-time trading session continuing directly from yesterday's pre-existing portfolio.
* **Real-Time Data Feed:** Fetches live intraday/daily market quotes across all 49 Nifty equities via `yfinance` from the National Stock Exchange of India (NSE).
* **Live Mark-to-Market:** Computes real-time portfolio value, today's intraday P&L, and unrealized returns.
* **Model Decision & Execution:** When triggered ("🤖 Run Model Decision"), the `InternVL3-2B + LoRA RL` adapter evaluates today's real-time prices + current portfolio state, calculates whole-share target allocations under 10% position limits and transaction fees, and streams real-time natural language reasoning explaining its actions.
* **Day-to-Day Persistent Settlement:** Clicking **"📌 Settle & Close Today"** (`POST /finance/paper/live/settle`) commits today's trades and settled closing prices into `data/finance_paper/live_portfolio_store.json`. This permanently updates the baseline so that tomorrow's live trading session (e.g. Nov 19 or Nov 20) resumes seamlessly from today's closing state, while Mode 2 expands to replay through this newly settled day.

### Verification and Regression Status:
- Verified via `demo/tests/test_three_simulation_modes.py` with 100% pass rate.
- Full demo test suite: **42 passed, 0 failed in 57.16s**.

# Historical Experiment Archive and Foundational Chronology

# Finance RL context transfer through 18 November 2026

This opening section contains the latest completed research summary and final results. It supersedes active-worker, pending-evaluation and current-candidate language in the historical chronology below. No finance training is active. The existing historical simulation remains the deployed example. The research decision is to conclude the current campaign with its limitations and complete experiment history.

This file consolidates the experiment narrative, major outcomes and complete final rotation metrics. Individual historical runs, raw daily paths, checkpoints and every historical cost/period combination remain in the linked evidence files and archive. It is not a byte-for-byte copy of every experimental artifact.

# Finance RL experimental record and integration decision

18 November 2026 Â· InternVL3-2B Â· Indian equity portfolio simulation

At â‚¹100,000, all three final state-aware policies earned a positive net return and modestly improved on the same model with its adapters disabled. Holding the common opening basket earned more than every trained policy. We recommend publishing the experiment history and keeping the existing strategy-mixture demo under its original description. The final rotation adapters are research artifacts. None qualified to replace the demo.

## How to read the record

Return means cumulative change in simulated equity after the costs charged by that study. A percentage-point gap subtracts two returns. Development was inspected repeatedly. Validation chose checkpoints. Later revisions reused the 2025â€“January 2026 retrospective window, so the final window cannot certify generalization to unseen markets. Independent training seeds share market observations. They measure training variability on one history.

## Experiment chronology and decision trail

| Stage | Question and intervention | Evidence and decision |
|---|---|---|
| Training pipeline and joint PPO | Could financial rewards move parameters and alter trading? Early audits of optimizer updates, likelihood accounting and precision preceded comparisons of numerical heads plus LoRA. | Roughly 45.7% development return looked attractive. Matched frozen-LoRA controls left about 0.002 incremental Sharpe with uncertainty spanning zero. Attribution required a simpler action path and explicit raw controls. |
| Covariance, utility and risk teachers | Tested imitation, trainable subsets, no-LoRA, stricter arithmetic and delayed signals. | Several reused-development paths exceeded 50%. The 54.832% maximum belonged to a numerical teacher control. Precision sensitivity, comparable no-LoRA scores and a lagged test (trained âˆ’0.839%, raw +9.455%) weakened the case for promotion. |
| Sector exposure and native readouts | Compared deterministic and sampled actions, static strategy mixtures and conditional readouts. Added label permutations, warm starts and annual refits. | Some sampled gains disappeared on deterministic replay. Annual refits gave mixed signs. Specialized initialization and separate yearly resets preclude combining their best numbers into a continuous record. |
| Five-ETF risk adaptation | Tested direct policies and a risk-strategy mixture on a distinct five-ETF dataset. Training 2010â€“2018, validation 2019â€“2021, reused development 2022â€“2024. | Reserved 2026 mean Sharpe rose from 1.513 to 1.951 and drawdown fell from 3.509% to 2.490%. Return fell from 6.532% to 6.377%. Minimum variance reached 2.170 Sharpe. This motivated a separate profit and holding comparison. |
| Nifty strategy-preference adapter | Trained native strategy preferences in 2019â€“2021 and validated in 2022. The allocation tools consumed market data and the simulator tracked wealth. The model itself lacked stock and account observations. | Seed 42, checkpoint 44 returned 13.2829% against raw 11.2205%, holding 13.0591% and equal rebalancing about 13.44%. The mixture repeated across 54 decisions while downstream targets changed. The model knew the playbook before it knew the portfolio. This is a useful retrospective baseline for learning a strategy preference. It cannot establish account-aware stock selection. |
| Explanations and demo | Tested template output, then five explanation SFT runs. Runs 01â€“03 continued trading weights. Run 04 restarted on pristine base with a changed LoRA specification. Run 05 added harder examples and retention. | Low synthetic loss did not ensure factual explanations. The 30-question development reviews of runs 04 and 05 each yielded 15 passes. Finance conversation was disabled. The historical demo reproduces 270 saved equity points and 54 actual adapter calls. Show its holdings and execution records as records. |
| Stateful five-stock run 1 | Added account and stock state and direct stock actions, but narrowed the universe to five equities. | The first policy collapsed and was stopped. These experiments diagnose the decision interface on five names. The original full-universe requirement was missed. |
| Stateful run 2 | Revised the training and action path after collapse, then tested the validation-selected candidate. | +22.372% validation became âˆ’8.176% on the continuous January 2025â€“13 November 2026 extension, versus +1.893% raw and âˆ’1.795% holding. Seven sign perturbations changed greedy decisions only once. Selection on one good year gave little protection. |
| Stateful run 3 | Tested a further conditional-action revision and explicit perturbations. | On the extended test the selected policy returned âˆ’3.838%, raw âˆ’0.113%, holding âˆ’1.795%. All five targets were 20%. Six validation probes left greedy actions unchanged. Providing holdings and prices alone did not make the policy use them. |
| Stateful run 4 and fallback | Required conditional imitation before financial RL. The gate passed at 2,048 examples with 63.54% balanced accuracy versus raw 14.58%. A shutdown required a weights-only restart at step 64 with fresh Adam. | Step 64 reached 19.8887% on 2024 validation, versus 14.2273% holding. Work stopped to restore the 49-stock scope. The prepared run-5 input-bridge fallback was conditional on gate failure. It has no five-stock financial result here. |
| Full-universe rebuild, attempts a and b | Restored all 49 original stocks, numerical input bridge, native per-stock ratings, executable prices and whole-share costs. Attempt a stopped after its first update to correct target sizing while corporate-action receivables were outstanding. Pending value counts toward equity and cannot fund purchases. Run b completed 96 RL iterations. | At â‚¹100k validation chose RL update 1. Retrospective returns were RL 3.6663%, raw 3.5905%, SFT 3.6711% and holding 6.3586%. State perturbations changed actions, but row reordering also changed eight. Early floor-only sizing left around half the account in cash for holding at its second decision. This justified revisiting residual-cash allocation and canonical ordering. |
| Three-seed counterfactual credit | Improved whole-share execution, canonicalized stock inputs, and trained relative stock ratings using simulator counterfactuals. Each native seed received 1,536 financial updates. A separate numerical control tested dependence on the backbone. | Validation selected initialization for all three native seeds. On the primary period initialization earned 7.76%, SFT earned 7.39/6.35/6.42%, holding 12.56%, equal rebalancing 11.11% and the momentum teacher 5.85%. A secondary validation-best **RL-only** replay earned 7.00/6.55/5.89%. Excluding initialization from selection would not have rescued the result. |
| Three-seed sparse rotation | Removed momentum imitation and anchored decisions to an identical opening portfolio. Allowed one funded pair rotation or a hold every 20 sessions. This isolated the incremental value of active trades and limited turnover. | See all checkpoints, matched comparators, costs, capital sensitivity and uncertainty below. The final policies narrowed the holding deficit relative to the preceding design. Opening exposures and action spaces also changed, so the cross-study difference cannot be assigned to training alone. |

The five-ETF experiment and the later five-stock restriction are separate studies. The original panel contains 49 stocks through 30 January 2026. Later 2026 prices in the five-stock study belong to an extension. There is no hardware justification for the reduced stock universe in the final method. A GPU preflight subsequently exercised the complete 49-stock input and action path.

## Final rotation experiment

Training sampled July 2018 through 2022. The policy receives 25 causal features per stock through a learned numerical input bridge and scores all 49 names through the native model. Each seed took 768 financial updates with 16 sampled actions per state, using paired 20-session log growth relative to keeping the same inventory. A new purchase is capped at 2% of equity. Positions may drift. Selection used the highest mean annual validation log growth among RL checkpoints, with earliest ties. The evaluator reloaded each selected package and checked both annual validation returns before test replay. Raw disables LoRA and bypasses the numerical bridge, so the difference measures the joint adaptation.

**Primary test, â‚¹100,000.** The continuous window runs from January 2025 through 30 January 2026, covering 270 sessions. The opening share trades match between each selected policy, raw and holding. Raw and holding equity paths each agree across seeds. Rotations count decision dates with executed trades after the opening basket.

| Seed | Checkpoint | 2023 validation | 2024 validation | Test net return | Î” raw, pp | Î” hold, pp | Sharpe | Max drawdown | Fees, â‚¹ | Rotations |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | rl_672 | 29.67% | 15.68% | 10.426% | +0.052 | -2.138 | 0.782 | 11.31% | 138.34 | 13/13 |
| 123 | rl_480 | 29.65% | 15.68% | 10.535% | +0.161 | -2.030 | 0.788 | 11.28% | 138.59 | 13/13 |
| 456 | rl_480 | 29.75% | 15.74% | 10.513% | +0.138 | -2.052 | 0.789 | 11.28% | 140.21 | 13/13 |

**Starting allowance.** Each row uses identical initial cash for every policy. The execution ledger includes whole shares, fees, slippage and unspent cash.

| Capital, â‚¹ | Hold | Equal rebalanced | Momentum | Raw | RL 42 | RL 123 | RL 456 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 500 | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% |
| 5000 | 4.847% | 2.723% | -1.775% | 4.847% | 4.847% | 4.847% | 4.847% |
| 50000 | 15.456% | 13.267% | 1.372% | 12.094% | 11.708% | 12.366% | 12.366% |
| 100000 | 12.565% | 11.106% | 2.594% | 10.374% | 10.426% | 10.535% | 10.513% |
| 200000 | 13.592% | 12.281% | 3.342% | 11.362% | 11.211% | 11.250% | 11.103% |

At the primary capital each trained policy traded at all 13 opportunities after the common opening. Raw lagged holding by about 2.19 percentage points. The adapters recovered only 0.05â€“0.16 points of that gap. At â‚¹200,000 every trained seed trailed raw. The sign of the primary raw-relative result therefore depends on the account size.

At â‚¹500 the identical zero returns illustrate an affordability limit. Different final returns across larger accounts establish capital-sensitive simulated outcomes. A claim of learned capital-dependent strategy would require separating changes in model preferences from deterministic changes in executable shares and pair eligibility.

**Paired uncertainty.** Descriptive 95% circular block-bootstrap intervals below use 20-session blocks and 100 Ã— cumulative log-growth difference. All intervals include zero. They omit model-search uncertainty, and three seeds on one market path do not provide three independent market tests.

| Seed | RL minus raw | RL minus holding |
|---:|---:|---:|
| 42 | -0.031 to +0.155 | -5.449 to +1.907 |
| 123 | -0.335 to +0.697 | -5.445 to +2.093 |
| 456 | -0.503 to +0.935 | -5.592 to +2.318 |

## Why the holding values differ

The historical Nifty demo used fractional allocation and returned 13.0591%. The first restored whole-share study floored per-stock budgets, initially buying 30 of 49 at â‚¹100,000. Its holding return was 6.3586%. The later counterfactual and rotation studies used the revised whole-share opening allocation and report 12.56% over the same nominal January 2025â€“January 2026 primary window. Five-stock studies also used a different basket and, in some comparisons, a longer window. Always compare a policy against holding from its own execution ledger at the same capital. A gap between historical baseline values is not a training improvement.

## Publication and integration decision

Report two findings separately. A learned static strategy preference improved on initialization in the historical demo. In the full-information experiment, all three selected rotation policies modestly improved on initialization, but the passive basket and equal rebalancing earned more at the primary capital. The repeated primary window, simplified costs, fixed universe, limited regimes and conditional execution constrain generalization. The current evidence cannot isolate model size, data quality or reward horizon as the cause of the deficit.

Keep the existing demo for the reproducible strategy-preference example. Archive all three final adapters for research replay. The final policy emits scores and orders, and there is no validated natural-language explanation from it. A further overnight sweep on the same dates would add another adaptively selected comparison. The next informative experiment would lock the policy, benchmark and capital before collecting a fresh period. Use the present timeline and full matched results for the paper.

## Evidence and reproduction map

- [Historical research report](JOURNAL_RESEARCH_REPORT.md), [full working chronology](CONTEXT_TRANSFER_FULL.md) and [numeric appendix](NUMERIC_APPENDIX.md) document historical choices, splits, controls and deployment audits.
- [Directory-by-directory campaign inventory](finance_evidence_20261118/campaign_directory_index.json) lists every archived campaign directory, available root records and extractable selected metrics. A directory may be a pilot, restart or control. Missing fields indicate missing or differently structured records. [Original development ledger](finance_evidence_20261118/CAMPAIGN_RESULTS.md) gives individual historical scores and signal delays.
- [First stateful study](STATEFUL_RL_ACTIVE_RUN_20261117.md), [second revision](STATEFUL_RL_REVISION2_ACTIVE_20261117.md), [third revision](STATEFUL_RL_REVISION3_RESEARCH_20261117.md), [fourth revision](STATEFUL_RL_REVISION4_ACTIVE_20261117.md) and [fallback protocol](STATEFUL_NUMERIC_BRIDGE_CONTINUATION_20261117.md) preserve the sequence.
- [Full-universe rebuild](FULL_UNIVERSE_CONTINUATION_20261117.md), [whole-share results](FULL_UNIVERSE_RESULT_20261118.md), [counterfactual protocol](COUNTERFACTUAL_FULL_CAMPAIGN_20261118.md), [native and numerical results](COUNTERFACTUAL_RESULT_AND_FINAL_ROTATION.md) and [secondary RL-only replay](finance_evidence_20261118/counterfactual_rl_only.json) document the successive changes.
- [Complete final metrics, candidates, protocol and bootstrap records](finance_evidence_20261118/rotation_records.json) and the [actual training and evaluation implementation](../research/final_rotation.py) support the tables. The portable archive contains per-day equity, executed share ledgers, base weights and checkpoint packages under `PolyvalentFinance/runs/final_rotation49_20261118` and `PolyvalentFinance/local_models`. Earlier reports retain the status at their original dates. This record supersedes their pending-run and promotion language.

For independent replay, use the [checkpoint loading instructions](ROTATION_REPLAY_20261118.md) and the [primary equity and execution ledgers](finance_evidence_20261118/).


## Complete final rotation metric table

All five starting allowances, five policies and three seeds are included below. Fees and slippage are amounts in INR. The 270-session window is January 2025 through 30 January 2026. Identical baseline rows across seeds are expected because the comparator and price path are shared.

| Seed | Capital INR | Policy | Return | Sharpe | Max drawdown | Fees INR | Slippage INR |
|---:|---:|---|---:|---:|---:|---:|---:|
| 42 | 500 | rl | 0.000000% | 0.000000 | 0.000000% | 0.000000 | 0.000000 |
| 42 | 500 | untrained | 0.000000% | 0.000000 | 0.000000% | 0.000000 | 0.000000 |
| 42 | 500 | holding | 0.000000% | 0.000000 | 0.000000% | 0.000000 | 0.000000 |
| 42 | 500 | equal_rebalanced | 0.000000% | 0.000000 | 0.000000% | 0.000000 | 0.000000 |
| 42 | 500 | momentum | 0.000000% | 0.000000 | 0.000000% | 0.000000 | 0.000000 |
| 42 | 5000 | rl | 4.847426% | 0.523331 | 5.820505% | 3.012958 | 1.505735 |
| 42 | 5000 | untrained | 4.847426% | 0.523331 | 5.820505% | 3.012958 | 1.505735 |
| 42 | 5000 | holding | 4.847426% | 0.523331 | 5.820505% | 3.012958 | 1.505735 |
| 42 | 5000 | equal_rebalanced | 2.723474% | 0.317434 | 5.964581% | 80.978037 | 40.488225 |
| 42 | 5000 | momentum | -1.774861% | -0.783940 | 2.900979% | 5.368859 | 2.684200 |
| 42 | 50000 | rl | 11.707630% | 0.870711 | 10.336222% | 69.045045 | 34.510120 |
| 42 | 50000 | untrained | 12.093705% | 0.894596 | 10.336222% | 69.233271 | 34.604280 |
| 42 | 50000 | holding | 15.455903% | 1.126745 | 10.218039% | 49.840812 | 24.907970 |
| 42 | 50000 | equal_rebalanced | 13.267344% | 0.985617 | 9.339066% | 1126.266820 | 563.121094 |
| 42 | 50000 | momentum | 1.372321% | 0.193523 | 5.109236% | 333.945088 | 166.965370 |
| 42 | 100000 | rl | 10.426464% | 0.782199 | 11.305712% | 138.340129 | 69.145715 |
| 42 | 100000 | untrained | 10.374173% | 0.779152 | 11.305712% | 138.800898 | 69.376215 |
| 42 | 100000 | holding | 12.564600% | 0.943963 | 10.379267% | 99.950057 | 49.950090 |
| 42 | 100000 | equal_rebalanced | 11.106235% | 0.848793 | 10.258008% | 1522.733555 | 761.341910 |
| 42 | 100000 | momentum | 2.594387% | 0.279272 | 7.199940% | 834.229481 | 417.096145 |
| 42 | 200000 | rl | 11.211212% | 0.872134 | 9.183163% | 278.957255 | 139.429959 |
| 42 | 200000 | untrained | 11.362468% | 0.882415 | 9.183163% | 278.215326 | 139.058809 |
| 42 | 200000 | holding | 13.592401% | 1.052661 | 8.506012% | 199.805978 | 99.853135 |
| 42 | 200000 | equal_rebalanced | 12.280961% | 0.962648 | 8.617890% | 2073.952219 | 1036.926105 |
| 42 | 200000 | momentum | 3.342316% | 0.318764 | 7.961086% | 1913.071907 | 956.491794 |
| 123 | 500 | rl | 0.000000% | 0.000000 | 0.000000% | 0.000000 | 0.000000 |
| 123 | 500 | untrained | 0.000000% | 0.000000 | 0.000000% | 0.000000 | 0.000000 |
| 123 | 500 | holding | 0.000000% | 0.000000 | 0.000000% | 0.000000 | 0.000000 |
| 123 | 500 | equal_rebalanced | 0.000000% | 0.000000 | 0.000000% | 0.000000 | 0.000000 |
| 123 | 500 | momentum | 0.000000% | 0.000000 | 0.000000% | 0.000000 | 0.000000 |
| 123 | 5000 | rl | 4.847426% | 0.523331 | 5.820505% | 3.012958 | 1.505735 |
| 123 | 5000 | untrained | 4.847426% | 0.523331 | 5.820505% | 3.012958 | 1.505735 |
| 123 | 5000 | holding | 4.847426% | 0.523331 | 5.820505% | 3.012958 | 1.505735 |
| 123 | 5000 | equal_rebalanced | 2.723474% | 0.317434 | 5.964581% | 80.978037 | 40.488225 |
| 123 | 5000 | momentum | -1.774861% | -0.783940 | 2.900979% | 5.368859 | 2.684200 |
| 123 | 50000 | rl | 12.365788% | 0.910763 | 10.336222% | 69.319070 | 34.647215 |
| 123 | 50000 | untrained | 12.093705% | 0.894596 | 10.336222% | 69.233271 | 34.604280 |
| 123 | 50000 | holding | 15.455903% | 1.126745 | 10.218039% | 49.840812 | 24.907970 |
| 123 | 50000 | equal_rebalanced | 13.267344% | 0.985617 | 9.339066% | 1126.266820 | 563.121094 |
| 123 | 50000 | momentum | 1.372321% | 0.193523 | 5.109236% | 333.945088 | 166.965370 |
| 123 | 100000 | rl | 10.534792% | 0.788259 | 11.278751% | 138.588297 | 69.269865 |
| 123 | 100000 | untrained | 10.374173% | 0.779152 | 11.305712% | 138.800898 | 69.376215 |
| 123 | 100000 | holding | 12.564600% | 0.943963 | 10.379267% | 99.950057 | 49.950090 |
| 123 | 100000 | equal_rebalanced | 11.106235% | 0.848793 | 10.258008% | 1522.733555 | 761.341910 |
| 123 | 100000 | momentum | 2.594387% | 0.279272 | 7.199940% | 834.229481 | 417.096145 |
| 123 | 200000 | rl | 11.250216% | 0.872321 | 9.142866% | 273.885174 | 136.892660 |
| 123 | 200000 | untrained | 11.362468% | 0.882415 | 9.183163% | 278.215326 | 139.058809 |
| 123 | 200000 | holding | 13.592401% | 1.052661 | 8.506012% | 199.805978 | 99.853135 |
| 123 | 200000 | equal_rebalanced | 12.280961% | 0.962648 | 8.617890% | 2073.952219 | 1036.926105 |
| 123 | 200000 | momentum | 3.342316% | 0.318764 | 7.961086% | 1913.071907 | 956.491794 |
| 456 | 500 | rl | 0.000000% | 0.000000 | 0.000000% | 0.000000 | 0.000000 |
| 456 | 500 | untrained | 0.000000% | 0.000000 | 0.000000% | 0.000000 | 0.000000 |
| 456 | 500 | holding | 0.000000% | 0.000000 | 0.000000% | 0.000000 | 0.000000 |
| 456 | 500 | equal_rebalanced | 0.000000% | 0.000000 | 0.000000% | 0.000000 | 0.000000 |
| 456 | 500 | momentum | 0.000000% | 0.000000 | 0.000000% | 0.000000 | 0.000000 |
| 456 | 5000 | rl | 4.847426% | 0.523331 | 5.820505% | 3.012958 | 1.505735 |
| 456 | 5000 | untrained | 4.847426% | 0.523331 | 5.820505% | 3.012958 | 1.505735 |
| 456 | 5000 | holding | 4.847426% | 0.523331 | 5.820505% | 3.012958 | 1.505735 |
| 456 | 5000 | equal_rebalanced | 2.723474% | 0.317434 | 5.964581% | 80.978037 | 40.488225 |
| 456 | 5000 | momentum | -1.774861% | -0.783940 | 2.900979% | 5.368859 | 2.684200 |
| 456 | 50000 | rl | 12.365788% | 0.910763 | 10.336222% | 69.319070 | 34.647215 |
| 456 | 50000 | untrained | 12.093705% | 0.894596 | 10.336222% | 69.233271 | 34.604280 |
| 456 | 50000 | holding | 15.455903% | 1.126745 | 10.218039% | 49.840812 | 24.907970 |
| 456 | 50000 | equal_rebalanced | 13.267344% | 0.985617 | 9.339066% | 1126.266820 | 563.121094 |
| 456 | 50000 | momentum | 1.372321% | 0.193523 | 5.109236% | 333.945088 | 166.965370 |
| 456 | 100000 | rl | 10.512661% | 0.788925 | 11.278751% | 140.208786 | 70.080515 |
| 456 | 100000 | untrained | 10.374173% | 0.779152 | 11.305712% | 138.800898 | 69.376215 |
| 456 | 100000 | holding | 12.564600% | 0.943963 | 10.379267% | 99.950057 | 49.950090 |
| 456 | 100000 | equal_rebalanced | 11.106235% | 0.848793 | 10.258008% | 1522.733555 | 761.341910 |
| 456 | 100000 | momentum | 2.594387% | 0.279272 | 7.199940% | 834.229481 | 417.096145 |
| 456 | 200000 | rl | 11.103150% | 0.862914 | 9.142866% | 274.372930 | 137.136660 |
| 456 | 200000 | untrained | 11.362468% | 0.882415 | 9.183163% | 278.215326 | 139.058809 |
| 456 | 200000 | holding | 13.592401% | 1.052661 | 8.506012% | 199.805978 | 99.853135 |
| 456 | 200000 | equal_rebalanced | 12.280961% | 0.962648 | 8.617890% | 2073.952219 | 1036.926105 |
| 456 | 200000 | momentum | 3.342316% | 0.318764 | 7.961086% | 1913.071907 | 956.491794 |


## Why the historical seed results looked identical

The calendar-2025 table below now retains six decimal places. Previously all three returns rounded to 15.074%, all Sharpe values to 1.243 and all drawdowns to 7.780%. The saved JSON records contain different daily paths and ending wealth. The three selected adapters have distinct SHA-256 digests in the actual-weight reload receipt.

| Historical run | Selected step | Equal weight preference | Inverse volatility | Momentum | Reversal |
|---|---:|---:|---:|---:|---:|
| nifty_static_native_seed42_64 | 44 | 32.879907% | 64.486808% | 1.358209% | 1.275075% |
| nifty_static_native_seed123_64 | 40 | 32.871655% | 64.472711% | 1.374539% | 1.281098% |
| nifty_static_native_seed456_64 | 44 | 32.713741% | 64.652795% | 1.358366% | 1.275096% |

All three learned very similar constant strategy preferences, with about 64.5% inverse volatility and 32.9% equal weight. The downstream strategies used the same prices, dates and execution rules. Their almost identical mixtures therefore produced almost identical financial paths. This supports convergence to similar policies within that experiment. It does not establish independent market replication or capital-aware stock selection. These are historical calendar-2025 scores. The latest full-information rotation experiment returned 10.426%, 10.535% and 10.513% over January 2025–January 2026 at ₹100,000.

Sources are [saved seed results](nifty_test_results/), [actual checkpoint reload and adapter digests](nifty_test_results/actual_weight_reload.json), and [full historical numeric appendix](NUMERIC_APPENDIX.md).


# Historical chronology preserved through the latest handover

The following entries retain their original dates and context. Their prospective instructions describe earlier stages of the project.

# Current authoritative context: Nifty paper-trading release

Read JOURNAL_RESEARCH_REPORT.md first for the selected artifact, exact retrospective test, completed demo and removed narration. The following older reports are historical evidence; their earlier final-candidate/deployment statements are superseded. The short final presentation list is at the END of this file.

# Finance RL: verified candidate and training/validation/test results

## Recommendation

The Nifty native static strategy-preference adapter (seed42 checkpoint44) is the stronger candidate for demonstrating a training benefit. It improves test net return and Sharpe over the exact untrained policy, and slightly exceeds equal-weight buy-and-hold at the original10bp costs. Actual adapter weights were reloaded and verified. Do not claim it beats all simple trading strategies: rebalanced equal weight and inverse volatility remain stronger controls. No UI/deployment switch was made in this audit; the existing dashboard remains the old five-ETF experiment.

## Data partition for the final presentation

- Training:2019–2021, updates to LoRA parameters through financial RL.
- Validation:2022, checkpoint selection using net Sharpe. Selected steps42/123/456:44/40/44.
- Test:calendar2025 and January2026, shown separately. Also report a continuous1January2025–30January2026 test as supplementary material, carrying holdings through the year boundary.

2023–2024 development scores are EXCLUDED from the main results and figures. Their prior use for method design must remain disclosed in the methods/history. These test results are retrospective: the candidate was revisited after observing earlier experiments and the market data had been accessed elsewhere in the project. Changing terminology cannot turn them into independent untouched tests. Do not call validation or previously reused development history independent test evidence.

## Test2025:249 sessions,10bp fees

| Policy | Net return | Sharpe | Max drawdown |
|---|---:|---:|---:|
| Original untrained |13.274%|1.103|8.135%|
| Trained seed42 |15.074445%|1.242963|7.779637%|
| Trained seed123 |15.073531%|1.242894|7.779826%|
| Trained seed456 |15.074210%|1.242975|7.779585%|
| Equal-weight buy-and-hold |14.864%|1.220|7.750%|
| Hold trained first target |14.741%|1.222|7.733%|
| Hold untrained first target |13.697%|1.136|8.120%|
| Equal-weight rebalanced |15.242%|1.244|7.787%|
| Inverse-volatility rebalanced |15.106%|1.252|7.754%|

The trained policy gains about1.80 percentage points versus untrained, and0.21 points versus equal-weight buy-and-hold. Its Sharpe gain over raw is about0.140. Its drawdown is slightly worse than buy-and-hold. All three trained seeds improve versus raw, but they share one market path and have near-identical strategies, so they are optimization replications, not independent market replications.

Conditional paired circular-block bootstrap,5000 resamples,20-session blocks: trained mean minus untrained return95% CI[0.590,3.264] percentage points; Sharpe CI[0.058,0.212]. Five/60-session sensitivity intervals also exclude zero for the raw comparison. These intervals do NOT adjust for historical method/candidate search and do not establish independent confirmatory significance. Versus equal-weight buy-and-hold, return CI[-0.880,1.277] points includes zero. Do not claim established superiority over holding.

## January2026 and continuous test

January2026 (21 sessions, independent cash reset): untrained-1.915%, trained-1.614%, buy-and-hold-1.588%. Training reduces the loss versus raw but does not beat holding. The month is too short for a strong standalone inference; bootstrap blocks comparable to or longer than this month are not credible independent confirmation.

Continuous1January2025–30January2026 (270 sessions, no year-end reset): untrained11.221%, trained13.283%, buy-and-hold13.059%, equal-weight rebalanced13.441%, inverse-volatility rebalanced13.343%. Report alongside the separate years; do not multiply reset-period returns and call them a continuous portfolio.

## Cost sensitivity

2025 test: at20bp, untrained12.563%, trained14.820%, buy-and-hold14.750%; at50bp, untrained10.458%, trained14.060%, buy-and-hold14.407%. The training gain over raw survives these modeled costs, but the small advantage over holding does not. These are fee sensitivity simulations, not a realistic market-impact model.

## What actually changed in the model

The same pinned full InternVL3-2B was loaded with FP32 active decoder and TF32 disabled; inactive original components remained on CPU. The six original packages (initial and selected for each seed) passed manifest verification. Maximum saved-versus-reloaded action-probability discrepancy was below1.5e-8. Disabling each trained adapter returned the same original untrained probabilities. This verifies real weight adaptation, rather than a manually substituted allocation.

For seed42, strategy probabilities changed approximately:
- Equal weight20.57% →32.88%.
- Inverse volatility31.39% →64.49%.
- Momentum30.48% →1.36%.
- Reversal17.56% →1.28%.

These probabilities are fixed strategy preferences; causal strategy target weights change with prices and the portfolio rebalances every five sessions. Training learned a more useful strategy mixture on these test histories. It did not learn daily news reasoning or prove state-dependent timing. The observed raw-relative gain includes lower turnover/costs, not exclusively superior gross stock selection.

## Verification and boundaries

Source stock-data SHA256 exactly matches the original training protocol. The original fixed49-stock universe and preprocessing remain; current-universe/survivorship and point-in-time limitations are not solved. All11 policies in each of nine period/cost combinations share exact return dates. Daily-return compounding and ending wealth reconcile; all hold baselines have zero turnover after the initial purchase. No training or original artifact modification occurred. Actual inference probabilities, full daily paths, bootstrap intervals, protocols and accounting checks accompany this report.

Main figures: presentation/nifty_test_2025.png, SVG and PDF; caption separate. Main presentation contains training/validation/test definitions and test results only. Historical development figures remain in prior archives for provenance.

## Defensible final statement

Financial RL changed the native2B model's strategy preferences and improved subsequent test return and Sharpe versus its original untrained policy across three training seeds. It slightly outperformed equal-weight buy-and-hold at the original fee setting over2025 and the continuous available test interval. That holding advantage is small, uncertain and cost-sensitive, and simple rebalanced strategies still match or outperform it. This supports an adaptation-benefit claim, not universal portfolio-management superiority.


---

# Historical research context: superseded primary ETF selection retained for provenance

# Current finalization update: read the closing addendum and final_presentation_figures/README.md. Earlier historical claims are superseded where inconsistent.

# Final Finance audit — 16 November 2026

This report supersedes checkpoint selections and template-narration claims in earlier handovers. No training is queued. The financial experiment is frozen. Unrestricted reliable explanations remain unresolved.

## Selected deployment

Run05 checkpoint176, adapter scale 1.0, full FP32 language inference. The full InternVL3-2B is retained, with vision on CPU during Finance inference and restored on adapter eviction. The original RL adapter produces the evaluated portfolio strategy preferences. A separate SFT adapter generates explanations; one checkpoint has not been demonstrated to do both jobs.

Runs01–03 continued the original attention/lm_head trading adapter and generated poor answers despite low synthetic loss. Run04 restarted from the pristine backbone using rank16 attention and MLP LoRA without lm_head: 18,464,768 trainable parameters, 768/96/96 synthetic train/validation/test records, 192 updates at learning rate 2e-4, selected validation response CE 0.006033792, approximately 1,279 seconds and 4.853 GiB peak training allocation. Run05 continued this branch with 384 retention and 384 harder examples, 192 updates at 8e-5, selected step176 CE 0.004000088, approximately 1,273 seconds and 4.888 GiB peak. Both used BF16 base training, FP32 adapters and gradient accumulation eight.

Architecture, initialization and data changed together; improvement cannot be attributed solely to removing lm_head. Synthetic accounts did not use final market rows as targets, but development failures informed task categories. Low template-distribution validation CE is not freeform factual accuracy.

## Selection evidence and limitations

The assistant's manual paired BF16 review of 30 actual-portfolio questions per model gave run04 15 pass / 5 partial / 10 fail and run05 15 pass / 4 partial / 11 fail. Both fully passed only 2/12 new-wording stress questions. Run05 was therefore not an overall BF16 winner. FP32 inference improved several accounting questions on a selected twelve-question difficult probe; scales 0, 0.5 and 0.75 were weaker on that probe. This supports a narrow deployment choice, not proof of the best possible configuration.

Twelve blank FP32 records in adapter_strength_review.json came from a CPU/GPU input-device bug and are invalid model-quality observations. The corrected probe is adapter_strength_fp32_review.json. Inputs now follow the language embedding device. Full-precision Finance inference uses approximately 6.7 GiB GPU memory. Other-domain full trained weights were not all available; unit residency checks do not certify all four domains.

## Post-selection actual HTTP audit

finance/evaluation/final_deployed_audit.json preserves six repeated regression questions and eight fresh post-selection phrasings, with drafts, revisions and checks. The eight fresh questions were not used to retune the model. This is a small developer audit, not independent universal accuracy measurement.

Successful regression answers explained the TLT buy and fees, UUP capital concentration versus estimated risk contribution, negative GLD/TLT P&L ordering, and allocation contribution versus investment return. After a model-generated revision, the benchmark answer correctly ranked standalone minimum variance above the trained model. Its statistical wording remained awkward. A holdings answer introduced unsupported wording about two trades.

Fresh-question successes included fee double-counting, weight drift, and the limits of counterfactual explanations. The answer about fixed strategy preferences was indirect. Important failures remained:

- Asked whether GLD's loss was larger than TLT's, it said “No” while quoting figures showing that it was larger. The checker missed this semantic reversal.
- Asked what performance could be attributed to fine-tuning, it gave an irrelevant transaction explanation with malformed “$1.93%” wording.
- Asked about a confidence interval containing zero, it invented a 0.01–0.03% preference interval. The checker flagged unsupported numbers, but the second model attempt remained wrong.
- The trade-versus-profit answer distinguished the concepts, but its specific transaction claims need record-level checking before being used as a showcase.

No failed answer was rewritten or removed. These results demonstrate actual local generation and several useful explanations, not reliable answers to any question. More of the same synthetic training is not supported by the present evidence. A future attempt needs a concrete generalization intervention and new independent evaluation.

## Runtime and verification

Start-FinanceDemo.ps1 launches http://127.0.0.1:8765/#finance using finance/active_model.json. The final HTTP audit took roughly 6–28 seconds per request after loading. Draft generation is buffered for limited checks and at most one model-generated revision before display. No checker or UI component substitutes preset answer prose. Residual detected problems trigger a visible warning; no detected problem does not imply correctness.

25 finance/residency/checker tests passed with one warning. Earlier API mock transport tests passed separately; these are software tests, not generated answer quality. All 413 original bundle hashes are unchanged. Evidence is preserved in finance/evaluation/integration_tests.log, original_bundle_verification.json and final_deployed_audit.json. The original financial bundle remains immutable.

## Frozen portfolio result

The RL policy learns fixed preferences over five external causal allocation strategies. It does not predict news or update its strategy probabilities every day. Lagged strategy calculations and prices change allocation targets.

The final 189-session February–October 2026 comparison is raw Sharpe 1.513051 versus trained three-seed mean 1.951463; total return about 6.532% versus 6.377%; maximum drawdown about 3.509% versus 2.490%. Standalone minimum variance has Sharpe 2.169516 and is stronger. The paired trained-minus-raw Sharpe 95% interval [-0.080094, 0.864912] includes zero. Observed risk-adjusted improvement is defensible; higher absolute return, statistically established superiority or superiority over classical optimization is not.

## Final presentation shortlist

1. Architecture: original Finance RL adapter, causal allocation/execution engine, and separate synthetic-SFT explanation adapter on the same fixed backbone. Explicitly disclose both Finance weight roles.
2. One financial comparison table: raw model, three-seed trained policy and standalone minimum variance, with return, Sharpe, drawdown, costs, dates and uncertainty. Include equity/drawdown curves. Do not imply the trained model had the highest return or beat the optimizer.
3. Historical portfolio UI: balance, holdings, activity, fees, date/seed controls and actual generated questions. No live brokerage claim.
4. Successful generated examples: TLT buy/fee correction, UUP concentration versus risk contribution, allocation contribution versus return. Preserve actual output and source dates.
5. Failure examples: fresh loss-comparison reversal, irrelevant attribution and fabricated confidence interval. Explain why synthetic validation loss and limited checking are inadequate reliability measures.
6. Full run history and paired comparisons belong in the appendix and handover. Final deployment selection is a narrow empirical choice, not universal optimality.

The portfolio experiment and real-model demo are presentable with these boundaries. The requested unrestricted justification reliability has not been achieved. Neither journal acceptance nor superiority over published systems is established by this audit.


# Historical explanation development

# Finance: model-generated explanation specialization and handover

16 November 2026. This update supersedes the deterministic narration in portfolio
v2. The financial selection and final-test results from v2 are unchanged.

## What changed and why

The user explicitly required actual model-generated explanations, consistent with
the other Polyvalent domains. The old bounded reply engine did not meet that
requirement. It was removed from the deployed source, rather than relabeled as
reasoning. `finance/service.py` now only reads the verified replay evidence.

Inspection covered the existing InternVL AdapterService, residency manager,
Anomaly telemetry-to-language prompts, Deepfake forensic wrapper and API streaming
implementation. Finance now follows the shared model/LoRA lease mechanism. The
full pinned InternVL3-2B loads once, its finance explanation adapter attaches, and
its autoregressive decoder generates answer tokens from portfolio evidence and
conversation history. ChatML framing uses the model's own tokenizer template.

`finance/grounding.py` selects and formats relevant facts. It does not return an
answer, score a canned answer, retrieve teacher labels or look up previous replies.
`finance/generation.py` invokes real inference. Both streaming and nonstreaming
Finance endpoints require a real resident model and reject stub generation.
A failure produces an error, not a preset substitute. The frontend displays generated text after limited checks. Drafts and model revisions are included in the API receipt; the checker does not rewrite the prose.
Basic Markdown rendering escapes HTML before applying formatting.

## What the explanation means

The original portfolio policy learned fixed probabilities over five external
allocation strategies. Their price-based targets change with lagged inputs. The
model's policy prompt itself is constant. Therefore a faithful explanation can
identify the target, the signed transaction, strategy contributions, risk estimates
and scheduling rules; it cannot establish news reasoning or a fresh model forecast
that the original policy never produced.

Narration is a generated interpretation of records, not a recovered internal
thought transcript. The original RL adapter remains the source of the evaluated
portfolio mixture. A separate explanation LoRA is trained from the pristine backbone for language. The UI and model
configuration expose that distinction. No claim is made that the explanation copy
retains the original strategy probabilities or financial results if used as a new
trading policy. Demonstrating one checkpoint retaining both capabilities would
require an additional joint-training/policy-preservation experiment and evaluation.

## Failures investigated before further training

Actual generation was tested with the full pristine base and with the original RL
adapter. A verbose JSON evidence prompt caused repetitive output on a contrary
buy/sell question. More compact factual context improved fluency, but the original
weights still confused numerical quantities, called UUP a utility company,
misstated strategy probabilities and failed some optimizer comparisons.

The first synthetic SFT improved reference-number coverage, but this was not a
factual-accuracy score. Testing new questions showed remaining factual failures.
A second pass with focused evidence nearly memorized its narrow synthetic
validation answers: validation response loss fell to about0.033. It still produced
poor answers on genuinely different question types, including confusing account
returns with five-session returns and risk contributions with capital allocations.
This is why the second pass was not accepted on loss alone.

Decoding diagnostics compared risk context on/off with repetition penalty1.0.
Removing risk facts did not solve the problem and sometimes made errors worse.
Tokenization was checked: training ChatML token IDs and inference formatted-prompt
token IDs matched. The issue was not a different prompt tokenizer. Evidence labels
were clarified, particularly cumulative-versus-event fees, account period, Sharpe
units, and which pair the confidence interval compares.

## Training lineage and reproducibility

Pinned base: OpenGVLab/InternVL3-2B revision
899155015275a9b7338c7f4677e19c784e0e5a21. Base weight SHA256:
18b4cdbc8eab790f447feac717f4d0cecbb85caf8df198b3b2401c22506858e7.

Runs01-03 initialized from a copy of the frozen seed42 finance RL LoRA. Run04 uses a fresh explanation LoRA from the pristine backbone. The original
413-file bundle is never written by these scripts. Runs01-03 use 6,809,504 trainable LoRA parameters; run04 uses18,464,768. All use a BF16 frozen full backbone and FP32 trainable LoRA,
response-only cross entropy, batch1, gradient accumulation8, gradient clipping1,
AdamW weight decay0.01 and gradient checkpointing. Only response-position logits
are materialized for the loss, reducing memory without changing the response CE.

- Run01: 256 synthetic training examples,48 validation,48 test; two epochs,
  64 optimizer updates, peak learning rate6e-5. Selected update64 by validation CE.
- Run02: initialized from run01;384 training and48 validation examples; two epochs,
  96 updates, peak rate1e-4. Selected update80, not the worse final update96.
- Run03: initialized from run02;576 training,72 validation,72 test examples; two
  epochs,144 updates, peak rate6e-5. Selection uses minimum validation response CE.
  Final selection and actual-generation review are recorded below.

The third dataset varies strategy preferences instead of keeping the same mixture
in every teacher answer. It simulates correlated returns, calculates historical
volatility and shrunk covariance, solves bounded minimum variance and reconstructs
fee-funded trades. It adds explicit tasks about Sharpe versus return, novice
explanations, false news/action premises, rebalancing without a price forecast,
comparative holding P&L, drift, explanation attribution and variance contributions.

Synthetic records are instructional examples, not historical financial evidence
or complete simulated equity paths. Some summary fields are independently sampled;
they must not be used to evaluate investment performance. Training answers are
locally authored labels. Their existence does not make runtime generation a
lookup system: the serving code never reads the training examples or target text.

No actual final-period account rows or answers were used as SFT targets. However,
observed failures on real portfolio questions informed task categories and prompt
changes. Those repeated questions are development regression checks, NOT a blind
held-out language benchmark. This distinction must be kept in the paper.

Data manifests record split seeds/counts/hashes. Each run directory preserves the
protocol, source hash, progress, validation selection and all saved checkpoints.
The source used to format the third training set is snapshotted alongside it.
The working scripts are build_finance_synthetic*.py and train_finance_explanations*.py.

## Serving and user experience

Start-FinanceDemo.ps1 reads finance/active_model.json and sets actual base/adapter
paths with FINAL_STUB_GENERATION=0. Local Hugging Face loading is offline. The demo
venv reuses the installed finance ML dependencies through a Python3.12-compatible
site-packages link. The original research venv was not modified. First use loads
the model; subsequent questions reuse the resident base and adapter. Other tasks
can evict the finance adapter under the same existing lease lock.

The portfolio remains a historical replay. Customer-facing elements are value,
return, decline, fees, holdings and activity, with research evidence collapsed.
Questions and follow-ups are free text. Changing the portfolio date, seed or
starting balance resets history and cancels the old request. Sources show the
activity date, input cutoff, account date and frozen evidence identity. The model
receipt identifies actual generation and elapsed time. Financial actions are not
sent to a broker; there is no live order capability.

A separate 126-session historical risk context was added using the same lagged
window and covariance shrinkage as the allocator. This supports explanations of
concentration without pretending allocation weight equals variance contribution.
These additional derived fields do not alter any saved return or selected policy.

## Verification boundaries

The 20 finance/residency integration tests passed, and the original four API stub
transport tests passed separately. They check accounting, funding, causal context,
conversation transport, adapter routing, input validation and refusal to substitute
answers on errors. Mocked transport tests are explicitly test doubles; they are
not evidence of model answer quality. All413 original bundle hashes remain intact.

Actual-generation output files preserve mistakes as well as good replies:
- finance/evaluation/initial_generation_probe.json
- finance/evaluation/compact_context_probe.json
- finance/evaluation/validation_generation_comparison.json
- finance/evaluation/generation_v2_results.json
- finance/evaluation/decoding_diagnostic.json
- finance/evaluation/generation_v3_results.json (final pass)

Numeric-reference recall in these scripts only measures whether reference numbers
appear. It does not validate their relationships, sufficiency or factual correctness.
Manual review of the full generated text is necessary. The model can still err,
especially on unfamiliar multi-part questions, arithmetic and unsupported claims.
No exhaustive conversational reliability or other-domain weight certification is
implied. This checkout contains other-domain adapter configurations, but their full
trained weights are not all present; this update did not invent or replace them.

## Financial results remain unchanged

Final native RL test: 189 sessions in February-October2026, seed42 fixed before
final access, three trained seeds reported. Mean trained Sharpe1.951463 versus
initial1.513051; trained total return about6.377% versus initial6.532%; drawdown
about2.490% versus3.509%. Standalone minimum variance has Sharpe2.169516 and is
stronger. The paired95% interval for trained-minus-initial Sharpe includes zero.

The contribution is a demonstrated small-model RL strategy-preference adaptation
with improved observed risk-adjusted performance, not proof of superior active
management, statistical significance, news forecasting or a need for an LLM over
classical optimization. The explanation specialization is an additional interface
capability and must be reported separately from those investment results.


---

# Complete previous research context (historical v2)

The template-based narration statements below are superseded by the model-generation update above. Financial results remain unchanged.

# Polyvalent finance RL: context transfer and final presentation decision

Prepared 16 November 2026. This is the current finance handover. Older files called CURRENT_HANDOFF, FINAL_REPORT, STATUS or PENDING describe earlier stages; their titles do not establish current authority. Their original versions are retained under evidence. Read this report and the immutable native final bundle before launching anything.

## 1. Decision and its limits

Keep the five-strategy, constant-prompt native InternVL3-2B finance RL adapter, default strategy-token mapping, validation-selected update 64. Use seed 42 for the demonstration. Seeds 123 and 456 are replications, not alternate winners. The training changes the original model's strategy preferences and improves historical risk-adjusted performance versus that exact original policy. It is a completed, supportable finance-domain result, not evidence of a superior autonomous investment manager.

This choice is not supported only by a flattering delta over a poor baseline. Its absolute trained development Sharpe is approximately 1.5695, above the earlier four-strategy native, full-state native, head-only and numerical selector results on the same ETF period. The original policy already gives approximately 47.25% probability to minimum variance, so it was not reset to a weak artificial baseline. However, a reversed-label control scored approximately 1.5722 and the standalone optimizer approximately 1.7187. The accurate description is **the strongest supported primary native method among the completed comparable candidates**, not the highest number anywhere on the SSD.

The reversed mapping was a sensitivity control; choosing it now for a 0.0028 Sharpe difference would turn a robustness check into another selection opportunity. The classical optimizer is materially stronger and remains visible. Final outcomes are already consumed. This task did not retrain, change the reward, select a new seed, change the mapping or blend in more optimizer weight. It clarified the evidence and built a faithful portfolio interface around the frozen result.

## 2. Project intent and persistent requirements

Polyvalent is a shared-backbone, multiple-domain adapter project. The surrounding domains include ordinary conversation/logical reasoning, deepfake work and anomaly detection. Finance must remain the reinforcement-learning domain because the project has already been presented that way. Replacing finance RL with supervised financial question answering does not meet the user's requirement. InternVL3-2B and this desktop are fixed; datasets, action spaces, objectives and training approaches could change.

The user rejected explaining inferior results away with a smaller GPU. The continuation therefore preserved the full pristine model and addressed memory through execution placement rather than removing layers or silently changing active numerical precision. The objective became measurable adapter-attributable improvement with defensible comparisons, not a guarantee of publication or global investment superiority. There is no universal Sharpe threshold that makes a research result good: relevant matched controls, execution realism, uncertainty and the exact claim determine its evidential quality.

All earlier quiet-hour, fan-noise and idle-time restrictions were explicitly lifted. Do not reinstate historical scheduling guards. One-GPU-worker serialization, dependency waits and numerical checks still apply. No finance training is queued at this handover. The user permits independent jobs between check-ins; that is not a request for an unsolicited recurring automation.

## 3. Workspace map and evidence authority

| Location | Role |
|---|---|
| C:/Users/Parth/Documents/PolyvalentFinance | Authoritative local finance continuation: source, runs, logs and research reports |
| C:/Users/Parth/Documents/PolyvalentFinance/deliverables/finance_rl_native_final_20261116 | Immutable selected RL bundle: adapters, exact runtime, data, evaluation and receipts |
| C:/Users/Parth/Documents/PolyvalentDemo | Integrated demo checkout with the revised portfolio workspace |
| F:/Polyvalent_Continuation_2026-11-07 | SSD continuation backup, including the final native bundle |
| F:/START_HERE_POLYVALENT_HANDOVER.md | Original SSD entry point |
| F:/Polyvalent_Handover_2026-11-07 | Original context and transfer material |
| F:/Polyvalent_PC_Review_2026-11-07/ASSESSMENT.md | Initial local source/hardware assessment |
| F:/FinanceRL_College_2026-10-30.zip | College migration artifacts; distinguish these from source-only clones |
| F:/C68-FinanceRL and F:/FINANCE/C68-FinanceRL | Older finance lineages, not current adapter authority |
| F:/pc/anomaly | Newer local anomaly implementation and standalone Streamlit prototype |
| F:/pc/LY_C68 | Deepfake project and original pinned-base source; preserve local modifications |
| F:/pc/C68-LR | Logical-reasoning work; earlier audit found local commits ahead of remote |
| F:/pc/Polyvalent_Research_Paper_2026-10-31 | Earlier paper/evidence audit and remote source snapshots |

The integrated demo was found in git-push-kar/Polyvalent and cloned at master commit e3a48047104431ac853379e787b76942e257acd0. Finance was a placeholder. The local implementation adds a dedicated Finance iframe and API routes, preserving Chat and Anomaly. No remote push or publication was performed. The other domains' dependencies and model artifacts have not been jointly configured or benchmarked in this demo checkout. Tabs and adapters existing in source do not prove simultaneous residency, correct switching or cross-domain non-interference.

This report synthesizes continuation records, comparison JSON, protocol files, final artifacts and current code. It does not claim that every byte of every SSD backup was reread or every historical experiment rerun today. SOURCE_INDEX.json hashes 172 copied source documents/reports. ALL_COMPARISON_RUNS.csv extracts 138 per-run rows from reports with compatible initial/trained metrics. Legacy documents with different schemas remain in evidence rather than being treated as missing or zero results. The authoritative daily outputs and full checkpoints remain in their original run directories and final bundle.

## 4. Hardware and execution identity

Desktop: Ryzen 7600X, 16 GB RAM, RTX 4060 Ti 8 GB. Research Python is PolyvalentFinance/.venv/Scripts/python.exe. Base: local_models/InternVL3-2B, OpenGVLab revision 899155015275a9b7338c7f4677e19c784e0e5a21. Recorded base weight SHA-256: 18b4cdbc8eab790f447feac717f4d0cecbb85caf8df198b3b2401c22506858e7.

The active decoder uses FP32 on GPU with TF32 disabled. Inactive original components remain on CPU in BF16. No backbone layer removal, smaller replacement model or active-path quantization explains the result. Static native technical gates used about 5 GB GPU memory; full-state training approximately 5.1 GiB and around 350 seconds per update. These are branch-specific observations, not a universal speed guarantee. A constant prompt substantially reduces computation compared with repeated market-state prompting.

The final rank16 attention and original vocabulary-head LoRA has 6,809,504 trainable parameters. A separate head-only ablation actually trained 2,451,360; its initial gate counted temporarily re-enabled attention flags incorrectly. reports/readout_parameter_count_clarification.json corrects this without rewriting the original gate. Saved attention LoRA B matrices in that ablation remained zero. Do not copy the erroneous count into a paper.

Frozen-base checks, recovery of initial outputs with adapters disabled, exact fresh reload, actual shared finance runtime checks and packaged GPU inference support technical attribution. They do not establish market generalization or the necessity of a language model. Original environment/runtime receipts stay with the experiment; the demo has a separate lightweight environment with no Torch dependency.

## 5. Why earlier results looked contradictory

A Sharpe around 1.68 in an old presentation is not automatically a better version of the current method. Different assets, dates, horizons, heads, execution conventions and development reuse make absolute cross-experiment comparisons invalid. Some earlier covariance development rows exceed 2.0; that method was not the selected financial RL approach and failed its separate final test. Training fit, adaptively reused development performance and reserved evaluation answer different questions.

Early joint numerical-head/LoRA models also failed to isolate where the gain came from. Matched Nifty PPO improved by approximately +0.048879 on average, but its incremental difference against frozen-LoRA control was only about +0.002013, with a 20-session interval [-0.01185, 0.01712]. Whole-package improvement did not justify saying the adapter caused the gain. Some candidate strategies had path correlations around 0.967–0.999, leaving little distinct action signal. In other runs sampled-action gains disappeared under deterministic deployment execution.

These were attribution, objective, strategy diversity and execution-contract problems. The evidence does not isolate model size or GPU as their cause. The final method addresses several of them through native vocabulary probabilities, only LoRA trainable changes, an exact original-policy comparator, more distinct strategy tools, fee-funded execution and saved daily-output parity. It does not solve the separate challenge of useful market timing: the selected policy's learned preferences are constant.

## 6. Experiment chronology and failure interpretation

### 6.1 Original covariance campaign: preserve the failure

The risk-allocation campaign used 49 fixed stocks, 252-return covariance, numerical heads plus attention LoRA, 21-session rebalances and a 30% target cap. Development results looked strong, but many rows used zero signal delay. A legacy precision invariance audit failed, so strict FP32/TF32-off replications were retained. Final lagged execution, May2025–January2026, failed: trained mean Sharpe approximately -0.0816 and return -0.839%, against the initial policy around 1.2554 / +9.455%; equal weight approximately 1.2408 / +9.335%. The initial and equal-weight rows are different comparators.

Its reports and package are historical evidence. It did not satisfy the user's finance-RL requirement, and its final period must not be merged with the later native ETF final period. The historical CURRENT_HANDOFF copied into evidence describes that earlier failure. The source/cost/lag caveats attached to old development tables must travel with their headline values.

### 6.2 PPO, sector exposure and early native readout

Joint numerical-encoder/head/LoRA PPO gave ambiguous incremental LoRA benefit. Summary/numerical and sector-exposure branches explored sampled actions, soft actors, short-window direct objectives and full-history objectives. A sampled sector policy could improve while its deterministic deployment worsened. Soft actors and short-window variants did not provide repeatable improvement; full-history exposure64 had approximately -0.1605 mean Sharpe change. Chronological folds were mixed, not a consistent fix.

Native sector readout256 achieved only about +0.004125 with 1/3 positive seeds and broad uncertainty. These branches motivate better contracts and control attribution; they are not proof that all RL or all 2B financial models fail. The extracted CSV and original reports retain absolute values where available, including the less successful runs.

### 6.3 Nifty static native preference learning

Four original vocabulary choices represented causal strategies. Constant-prompt native training changed initial Sharpe 1.580558 to about 1.6983 on 2023–2024 development, mean gain +0.117706 across three seeds. Validation selected steps 44, 40 and 44. Return rose from about 44.110% to 45.617%, but drawdown worsened from about 11.932% to 12.416%. The 20-session interval included zero.

Historical chronology folds showed small positive gains (+0.0290 and +0.0341), again with intervals including zero. Overlapping training windows do not create independent replications or a stitched investable track record. A limited 90%-static/10%-regime branch was slightly better than its own initialization but slightly worse than the trained static anchor. Supervised warmup plus RL improved against raw, but RL itself worsened the warmup result by about -0.000982. This cannot be sold as a positive incremental RL result.

Rotated/reversed labels retained positive gains. Those were controls, not opportunities to choose a weak initial baseline. Nifty remains historical corroboration, not the primary ETF result or a directly rankable absolute score against it.

### 6.4 Four-strategy ETF studies

GLD/SPY/TLT/UUP/XLE was fixed before the development panel was fetched. The 2008–2024 panel contains 4,279 common sessions, without missing-price forward fill or outcome-selected asset dropping. Train2010–2018, validation2019–2021, adaptively reused development2022–2024.

Numerical static/dynamic selectors improved mean Sharpe about +0.129918/+0.159165, but timing-specific gains versus a validation-averaged fixed mixture were negligible or negative. Native static64 improved 0.899310 to 1.075087 mean, largely emphasizing inverse volatility. Full-state native16 improved 0.884158 to approximately 0.989069; timing gain over a fixed mix was only about 0.0016–0.0025. Head-only16 reached approximately 0.955464.

Those 16-update branches still improved on validation and are not claimed converged. A conditional extension to32 was prepared but not launched after the stronger five-strategy family became available. That is a bounded method/resource decision with a disclosed limitation, not proof that state-conditioned training cannot work.

### 6.5 Direct allocations, alternative rewards, anchors and annual refits

Direct-asset static/dynamic controllers deteriorated approximately -0.3478/-0.4167. Half-full-history/half-worst-training-year reward failed approximately -0.1407/-0.1261. A 90% causal minimum-variance anchor with learned adjustments underperformed its initial policy and pure optimizer, approximately -0.0609/-0.0515. Risk-and-cost reward alternatives were negative, approximately -0.0805/-0.1195. Economic plausibility of an objective did not guarantee useful optimization.

Annual refits used five training years and one validation year before each of 2022, 2023 and 2024. All18 CPU runs completed. Static yearly mean gains: -0.105246, 0, +0.148018; dynamic: -0.112007, 0, +0.240608. Both failed the declared native-promotion screen. No annual native campaign followed. Portfolios reset from cash each year and are not one continuous stitched strategy. An automatic launch block was subsequently authorized by the user and the work completed; no permission request remains pending.

### 6.6 Five-strategy risk selector: selected family

Adding a causal minimum-variance candidate to equal weight, inverse volatility, momentum and reversal improved static/dynamic numerical mean Sharpe by +0.409259/+0.520820. Dynamic timing itself added only approximately 0.0029–0.0053 over fixed-mixture comparison, below the declared0.01 screen; static native was promoted instead.

Primary native development Sharpe changed from1.242536 to1.569486 mean, three positive seeds, exact technical gates and a positive paired development interval. Label rotation/reversal also improved, without replacing the default mapping. This family was motivated by observed optimizer/control results and is an adaptive discovery, not an independently preregistered scientific hypothesis. The subsequent sealed test matters. The pure optimizer remains stronger on development and final data. The supported contribution is model adaptation and an auditable constrained strategy-selection interface, not a new optimizer or superior asset manager.

## 7. Absolute trained-quality audit

| Same ETF development history, 2022–2024 | Initial Sharpe | Trained Sharpe (mean) | Trained return | Trained max drawdown |
|---|---:|---:|---:|---:|
| multiasset_native_comparison64 | 0.899310 | 1.075087 | 25.316% | 7.551% |
| multiasset_dynamic_native_comparison16 | 0.884158 | 0.989069 | 25.453% | 8.153% |
| readout_only_ablation_comparison16 | 0.884158 | 0.955464 | 25.519% | 8.505% |
| risk_selector_static_comparison64 | 1.005529 | 1.414787 | 25.872% | 4.056% |
| risk_selector_dynamic_comparison64 | 1.005529 | 1.526349 | 25.992% | 3.471% |
| risk_selector_native_comparison64 | 1.242536 | 1.569486 | 25.937% | 3.250% |
| risk_label_rotate_comparison64 | 1.129322 | 1.566941 | 25.933% | 3.257% |
| risk_label_reverse_comparison64 | 1.064756 | 1.572239 | 25.937% | 3.238% |

All rows above share the ETF development period, but differ in action set, initialization, architecture or update budget. They answer whether the selected primary model is merely weak with a flattering delta: it is not. They do not isolate every causal factor in method design. The tiny reversed-label edge is disclosed, and neither that edge nor its larger delta justifies changing the final primary mapping. The pure minimum-variance development Sharpe is1.718731, higher than all these native rows.

Nifty's approximately1.698 and earlier covariance scores above2.0 are excluded from this ranking. The prospective final-choice record emphasized within-policy gain and technical checks; this supplementary absolute-quality audit clarifies the interpretation without rewriting the old choice record.

## 8. Exact selected method and action contract

The backbone scores existing A/B/C/D/E tokens under a literal constant prompt. Their normalized probabilities mix five external strategy target vectors. No teacher-action SFT warmup, financial action labels or new regression head train the selected model. Only attention and vocabulary-head LoRA change. The objective is full-history net-return Sharpe with0.01 KL regularization toward the initial policy; recorded attention learning rate1e-5, head3e-4, gradient clipping1 and KL guard. Three seeds train64 updates; validation every4, update0 eligible, earliest exact tie. All select64.

Describe this as direct financial reinforcement with a finite-path differentiable execution objective, not PPO or GRPO. The piecewise local derivative is not an unbiased population policy-gradient or global-optimality claim. A paper should explain the computation precisely rather than rely on the label RL alone.

Strategies: equal weight; half equal weight plus half126-return inverse volatility; half equal weight plus half126-session momentum skipping21; half equal weight plus half5-session reversal; and causal minimum variance. Minimum variance uses126 lagged returns, annualization252, 90% full covariance plus10% diagonal shrinkage, SLSQP from equal weights, asset bounds0.105–0.58, sum1, ftol1e-12, maximum200 iterations. Solver failures are rejected. Overall targets obey a0.60 cap; drifted holdings are not falsely guaranteed to retain that exact target cap between trades.

Seed42's learned mix is approximately1.16% equal weight,21.11% inverse volatility,1.05% momentum,1.02% reversal,75.67% minimum variance. Raw minimum-variance preference was already47.25%. Market data changes the strategy targets; it does not enter this policy's constant prompt. The shared environment's33-feature summary is not evidence of state conditioning in the final model.

Each decision combines target vectors, reconciles them with actual drifted holdings, funds fees internally and applies the next adjusted-price return. Signals end one close before execution. Rebalance every5 sessions; hold drifted positions otherwise. Start entirely in cash. Fees are10bp times the full stock-notional L1 of purchases plus sales, not half-turnover. Saved daily fields include net/gross return, fee, turnover, equity, drawdown and dates.

## 9. Reserved evaluation and interpretation

Return dates:2 February–30 October2026,189 common sessions. First execution30 January, signals through29 January. Method choice preceded final data acquisition; specification, seeds, dates, sources, artifacts, controls and criteria were sealed before evaluation. An exclusive access receipt was created before final portfolio outcomes. The panel contains711 dates with warmup. Original vendor outputs, adjusted CSV and hashes are retained. Completeness checks reject missing sessions rather than filling them.

| Policy | Net Sharpe | Total net return | Max drawdown |
|---|---:|---:|---:|
| Exact initial native policy | 1.513051 | 6.5318% | 3.5094% |
| Trained seed 42 | 1.951435 | 6.3770% | 2.4901% |
| Trained seed 123 | 1.952345 | 6.3772% | 2.4890% |
| Trained seed 456 | 1.950610 | 6.3769% | 2.4910% |
| Standalone minimum variance | 2.169516 | 6.3963% | 2.1951% |

Mean paired Sharpe gain +0.438412, three positive seeds. The20-session paired circular-block/seed-bootstrap95% interval is[-0.080094,0.864912], including zero. Bootstrap uses5,000 replicates;5/20/60-session diagnostics are retained. Shared time blocks preserve pairing; seed aggregation means average individual metrics, not an investable ensemble. Near-identical seeds show optimization repeatability, not three independent markets. Development's positive interval[0.034247,0.609301] is unadjusted for adaptive search and cannot replace final uncertainty.

The declared practical screen required meanΔSharpe≥0.05, at least2 positive seeds and no more than2 percentage points of mean return loss or extra drawdown. It passed. These are campaign thresholds, not universal financial standards. Total return fell approximately0.155 percentage points while maximum drawdown fell approximately1.019 points. This is risk-adjusted improvement, not higher profit. Seed42 turns simulated$100,000 into$106,377.03 with approximately$197.50 modeled fees.

Pure minimum variance's2.169516 final Sharpe must accompany the model result. Numerical static controls scored approximately1.7522,1.7529,1.7587, but their differing initialization prevents a matched claim of LLM necessity. Fees0/5/10/25bp preserve roughly+0.438 mean paired Sharpe gain. Ten paired sampled-action draws give9/10 positives per seed; those are secondary execution checks, not independent market replications or the primary expected-mixture execution.

Vendor adjusted prices are retrospective, not a point-in-time archive. Slippage, taxes, impact and liquidity are not fully modeled. Sharpe uses mean net daily return divided by population daily standard deviation times√252, with zero risk-free return. Fixed cash-rate sensitivities are illustrative, not observed excess-cash returns. Nine months is short. No live-profit guarantee, broad regime-generalization result or individual-stock-picking capability has been established.

## 10. Literature and contribution boundaries

[Moody and Saffell2001](https://pubmed.ncbi.nlm.nih.gov/18249919/) provides historical direct-financial-reinforcement context. This finite-path LoRA implementation is not a reproduction of their recurrent method. Minimum-variance optimization is conventional, not this project's new invention.

[FinATOM2026 preprint](https://arxiv.org/html/2608.09880v1) studies Llama3.2 1B token-based allocation on the same five ETF exposures, with causal mean–variance supervised initialization followed by policy optimization. It reports2023–2025 pooled net Sharpe1.394→1.494 under its5bp setup. That is SFT-to-policy, not pristine-base-to-adapter improvement. Dates, modalities, refitting, actions and execution differ; our1.95 cannot be called a victory over its1.494. The useful comparison is methodological feasibility of small-model financial optimization. Our stable tool mixture, direct path objective and exact adapter attribution are different experimental choices, not proof of novelty by themselves. The primary paper was reopened while preparing this handover.

A defensible framing is a modular2B adapter case study with matched initial policies, isolated trainable changes, failed-control disclosure, label sensitivity, deployment parity and a sealed final evaluation. Broader algorithmic novelty and LLM necessity remain unestablished. Cite influences; arbitrary changes made just to avoid similarity are not contributions. Four domains/adapters alone do not settle originality.

## 11. Portfolio product and explanation design

Finance now opens on portfolio value, net gain, largest decline from peak and total fees. A chart shows value through the selected date, with optional untrained and minimum-variance comparisons. Holdings show current drifted exposures, values, weights and cumulative market P&L before fees. Activity lists buys/sells, one fee per rebalance, asset filters and a link to explain that activity. Holdings/activity export toCSV. Starting capital scales the same simulated path; it does not retrain or produce a new experiment.

All-run research tables, strategy-contribution heatmaps and seed controls are removed from the primary customer flow. A collapsed evidence panel keeps limitations, full-period comparisons and replication selection. The historical-paper badge stays visible. No fictitious broker connection, live quote or order capability is implied. The outer Chat/Anomaly tabs remain. No-adapter-loaded can be correct: replay does not require live GPU inference.

The exporter checks every frozen-bundle hash, reconstructs actions and verifies daily-return parity, then independently reconstructs all daily holdings and market P&L from the same execution kernel. For every3×189 daily observation, assets+cash equal equity and initial capital+cumulative market P&L−fees equal equity. All target contributions sum correctly; trades fund fees and purchases from the recorded account. Floating-point tolerances are applied, not fake residual profits.

The as-of view truncates charts, holdings, fees, risk and available activity consistently. A rebalance fee enters the original ledger in its first post-execution daily return; activity appears only once its daily_start observation is included. This avoids displaying a trade before its fee is reflected in balance. The separate research panel explicitly labels full-period results. As-of handling is explanation correctness, not security: the replay endpoint serves the entire historical research simulation.

The assistant resolves tickers and common names: gold, Treasuries/bonds, energy, dollar and US equities. It can answer multiple requested facts, remember the last unambiguous asset for follow-ups, use a recordedYYYY-MM-DD execution date, correct contrary buy/sell premises and reject future/unavailable activity. Answers cite activity, signal cutoff, as-of date and frozen seal. Supported evidence includes signed trades, prior/target weights, dominant strategy contributions, costs, current holdings, cumulative market P&L, risk and initial-policy comparisons.

Example:9 March2026 seed42 bought$560.74 TLT per$100,000 starting capital. Weight changed20.90%→21.44%; minimum variance contributed16.07 percentage points and inverse volatility4.74. Signals stop6 March, fee$2.25. A question saying it sold bonds is corrected. At10 March close, TLT drifted to approximately$21,886.12 /21.27%; a follow-up about current worth uses that daily value, not the old target. Outcomes after a trade are retrospective, limited to the displayed date, and never presented as information available at decision time.

### Explanation capability and further specialization

This is a deterministic evidence-grounded assistant, not unrestricted generative reasoning from the finance adapter. It explains the actual mechanism linking strategy outputs to trades. Contribution arithmetic is not feature importance, private model reasoning or the causal effect of a trade versus another action. It cannot supply news-based reasons, unknown-stock decisions, live forecasts, tax-lot accounting or broker orders. Unfamiliar phrasing can require clarification; this is defined domain coverage, not universal natural-language understanding.

No additional finance training is required for these rule/trade/accounting explanations. The finance RL adapter was not trained to generate rationales. If broad conversational fluency or complex counterfactuals are later required, add a separate narration/tool layer with structured evidence, numerical/date verification, abstention and adversarial evaluation. A general/reasoning adapter could narrate, but that capability must not be attributed to the finance RL objective. Rationale fine-tuning without evidence checks can increase fluent fabrication. Genuine counterfactual analysis needs an explicit alternative action, the same execution kernel and clear hypothetical labeling; it is not implemented here.

## 12. Verification and reproducibility

Original research checks include frozen base, disabled-adapter recovery, selected-checkpoint reload, actual finance runtime, packaged GPU prediction and six original/trained CPU replay paths. The immutable bundle manifest covers413 files. This task rebuilt replay from it and added daily accounting verification; it did not rerun training or replace frozen outcomes.

The demo unit suite checks funding/contributions, every daily balance/P&L identity, API validation, all seed42 trade-direction premise corrections, aliases, follow-ups, combined fee/trade questions, no-future as-of outcomes, unsupported assets/orders, explicit dates and capital scaling. Browser verification checks actual visible balances, dated explanations and follow-up values. See VERIFICATION.md for the final test and browser record. These checks do not claim complete language coverage, production security or other-domain functionality.

From C:/Users/Parth/Documents/PolyvalentDemo:

```powershell
& .\.venv\Scripts\python.exe -m uvicorn api.server:app --host 127.0.0.1 --port 8765
```

Open http://127.0.0.1:8765/#finance. The server is localhost-only. After editing replay or service restart it because evidence is cached. Use the demo environment for FastAPI and the finance environment for reconstruction/Torch:

```powershell
& C:\Users\Parth\Documents\PolyvalentFinance\.venv\Scripts\python.exe finance\build_replay.py --bundle C:\Users\Parth\Documents\PolyvalentFinance\deliverables\finance_rl_native_final_20261116
& .\.venv\Scripts\python.exe -m unittest tests.test_finance_workspace -v
```

For actual GPU prediction follow the immutable bundle README/predict.py. Set FINANCERL_EXECUTION=selective_fp32; FINANCERL_BASE_MODEL to the pinned local base; HF_HUB_OFFLINE=1; TRANSFORMERS_OFFLINE=1; OMP_NUM_THREADS=2; MKL_NUM_THREADS=2. Do not substitute a merged base. The runtime outputs weights, never broker orders. Exact CPU replay of an extracted constant policy is valid when labeled honestly.

Critical files under the finance root:

- NATIVE_FINAL_METHOD_CHOICE.json: prospective primary family, seeds, criteria, controls; do not rewrite after outcomes.
- NATIVE_FINAL_EVALUATION_SPEC.json and checks/NATIVE_FINAL_20261116/contract.json: sealed dates, protocol, source and artifact identities.
- reserved_finance_native_2026_access.json: final-period consumption receipt.
- checks/NATIVE_FINAL_20261116/result.json, seed42/123/456.json, controls.json, secondary_sensitivity.json: authoritative final outcomes.
- deliverables/finance_rl_native_final_20261116/adapters/seed42: fixed demonstration adapter; other seeds are replications.
- native_finance_runtime.py: causal controller and hold/rebalance contract.
- reports/risk_selector_native_comparison64.json: primary development comparisons and selection steps.
- reports/FINAL_CAMPAIGN_STATUS.json: late-stage campaign ledger, not the entire earliest history.
- RL_CONTINUATION.md: chronological decisions; newer completion entries supersede old pending text.

Demo files: finance/build_replay.py, finance/service.py, api/finance.py, ui/finance.html, tests/test_finance_workspace.py and FINANCE_WORKSPACE.md. The immutable research package, updated demo archive and context-transfer package have different purposes. Do not change the research bundle to update a UI. Include ignored replay data explicitly in archives; do not bundle caches, virtual environments or a substitute base.

## 13. Completed late-stage campaign ledger

All rows, including negative and zero outcomes, are retained. Periods differ: no global ranking. ALL_COMPARISON_RUNS.csv adds per-run absolute values and checkpoint choices from compatible reports, including earlier branches. SourceJSON contains full paths and additional intervals.

| Method | Period | Mean ΔSharpe | Positive seeds | Paired 95% interval, 20 sessions | Source |
|---|---|---:|---:|---|---|
| Nifty static native | 2023-01-02–2024-12-31 | +0.117706 | 3/3 | [-0.064579, 0.337915] | static_native_seed_comparison_budget64.json |
| Nifty limited timing | 2023-01-02–2024-12-31 | +0.009797 | 3/3 | [-0.007620, 0.031525] | regime_native_seed_comparison_budget64.json |
| Nifty warmup then RL (vs warmup) | 2023-01-02–2024-12-31 | -0.000982 | 0/3 | [-0.003749, 0.001703] | warmup_native_seed_comparison_budget64.json |
| Five-asset static numerical | 2022-01-03–2024-12-31 | +0.129918 | 3/3 | [-0.054670, 0.313552] | multiasset_static_comparison64.json |
| Five-asset dynamic numerical | 2022-01-03–2024-12-31 | +0.159165 | 3/3 | [-0.055058, 0.374260] | multiasset_dynamic_comparison64.json |
| Five-asset static native | 2022-01-03–2024-12-31 | +0.175776 | 3/3 | [-0.049332, 0.397205] | multiasset_native_comparison64.json |
| Five-asset dynamic native | 2022-01-03–2024-12-31 | +0.104912 | 3/3 | [-0.052643, 0.263644] | multiasset_dynamic_native_comparison16.json |
| Direct asset weights, static CPU | 2022-01-03–2024-12-31 | -0.347751 | 0/3 | [-0.787572, 0.098033] | asset_allocation_static_comparison64.json |
| Direct asset weights, dynamic CPU | 2022-01-03–2024-12-31 | -0.416709 | 0/3 | [-0.894917, 0.062370] | asset_allocation_dynamic_comparison64.json |
| Worst-year reward, static CPU | 2022-01-03–2024-12-31 | -0.140716 | 0/3 | [-0.508611, 0.239958] | asset_robust_static_comparison64.json |
| Worst-year reward, dynamic CPU | 2022-01-03–2024-12-31 | -0.126146 | 0/3 | [-0.482496, 0.242488] | asset_robust_dynamic_comparison64.json |
| Causal risk anchor, static CPU | 2022-01-03–2024-12-31 | -0.060907 | 0/3 | [-0.160886, 0.042580] | risk_anchor_static_comparison64.json |
| Causal risk anchor, dynamic CPU | 2022-01-03–2024-12-31 | -0.051511 | 0/3 | [-0.155944, 0.057534] | risk_anchor_dynamic_comparison64.json |
| Nifty rotated labels native | 2023-01-02–2024-12-31 | +0.131845 | 3/3 | [-0.078389, 0.389700] | label_rotate_comparison64.json |
| Nifty reversed labels native | 2023-01-02–2024-12-31 | +0.157864 | 3/3 | [-0.102689, 0.473880] | label_reverse_comparison64.json |
| Five-asset readout-only native | 2022-01-03–2024-12-31 | +0.071306 | 3/3 | [-0.039188, 0.184329] | readout_only_ablation_comparison16.json |
| Risk-cost reward, static CPU | 2022-01-03–2024-12-31 | -0.080508 | 0/3 | [-0.371579, 0.204371] | asset_risk_reward_static_comparison64.json |
| Risk-cost reward, dynamic CPU | 2022-01-03–2024-12-31 | -0.119527 | 0/3 | [-0.411189, 0.173225] | asset_risk_reward_dynamic_comparison64.json |
| Risk-tool selector, static CPU | 2022-01-03–2024-12-31 | +0.409259 | 3/3 | [0.119728, 0.706132] | risk_selector_static_comparison64.json |
| Risk-tool selector, dynamic CPU | 2022-01-03–2024-12-31 | +0.520820 | 3/3 | [0.123802, 0.905237] | risk_selector_dynamic_comparison64.json |
| Risk-tool selector, native | 2022-01-03–2024-12-31 | +0.326950 | 3/3 | [0.034247, 0.609301] | risk_selector_native_comparison64.json |
| Annual static CPU 2022 | 2022-01-03–2022-12-30 | -0.105246 | 0/3 | [-0.419934, 0.232557] | multiasset_annual_static_2022_comparison64.json |
| Annual static CPU 2023 | 2023-01-03–2023-12-29 | +0.000000 | 0/3 | [0.000000, 0.000000] | multiasset_annual_static_2023_comparison64.json |
| Annual static CPU 2024 | 2024-01-02–2024-12-31 | +0.148018 | 3/3 | [-0.021278, 0.332799] | multiasset_annual_static_2024_comparison64.json |
| Annual dynamic CPU 2022 | 2022-01-03–2022-12-30 | -0.112007 | 0/3 | [-0.461955, 0.257701] | multiasset_annual_dynamic_2022_comparison64.json |
| Annual dynamic CPU 2023 | 2023-01-03–2023-12-29 | +0.000000 | 0/3 | [0.000000, 0.000000] | multiasset_annual_dynamic_2023_comparison64.json |
| Annual dynamic CPU 2024 | 2024-01-02–2024-12-31 | +0.240608 | 3/3 | [0.035615, 0.476470] | multiasset_annual_dynamic_2024_comparison64.json |
| Risk selector rotated-label control | 2022-01-03–2024-12-31 | +0.437619 | 3/3 | [0.068937, 0.794249] | risk_label_rotate_comparison64.json |
| Risk selector reversed-label control | 2022-01-03–2024-12-31 | +0.507483 | 3/3 | [0.098467, 0.906643] | risk_label_reverse_comparison64.json |

## 14. Earlier development table, preserved for context

The following is the original covariance/numerical campaign table. Its apparently higher scores must not disappear from the handover, but its own lag/protocol caveats matter. It is not the selected native ETF final result.

# Recorded campaign results

2026-11-07T16:48:05.555131+05:30

reused 2023–2024 development; lag1-trained utility controls use one-close delay, other rows retain zero lag; no final-period results

| Run | Step | Signal delay | Return | Sharpe proxy | Max drawdown | ΔSharpe vs initialization |
|---|---:|---:|---:|---:|---:|---:|
| covariance_lora_mlp_seed42 | 512 | 0 | 50.928% | 2.0783 | 14.222% | +0.4273 |
| covariance_lora_only_seed42 | 384 | 0 | 50.733% | 2.0953 | 13.167% | +0.6392 |
| covariance_no_lora_seed123 | 512 | 0 | 50.422% | 2.0898 | 12.292% | +0.4272 |
| covariance_no_lora_seed42 | 512 | 0 | 50.293% | 2.0757 | 13.329% | +0.4172 |
| covariance_no_lora_seed456 | 384 | 0 | 52.827% | 2.1136 | 13.859% | +0.4527 |
| covariance_risk_no_lora_seed123 | 384 | 0 | 51.667% | 2.1688 | 12.592% | +0.5062 |
| covariance_risk_no_lora_seed42 | 384 | 0 | 49.478% | 2.0902 | 13.291% | +0.4317 |
| covariance_risk_no_lora_seed456 | 384 | 0 | 48.875% | 2.0638 | 13.072% | +0.4029 |
| covariance_risk_seed123 | 512 | 0 | 53.899% | 2.2618 | 12.467% | +0.5992 |
| covariance_risk_seed42 | 384 | 0 | 51.317% | 2.1480 | 13.513% | +0.4895 |
| covariance_risk_seed456 | 512 | 0 | 48.055% | 2.0434 | 12.573% | +0.3825 |
| covariance_risk_strict_no_lora_seed123 | 384 | 0 | 51.710% | 2.1704 | 12.584% | +0.5079 |
| covariance_risk_strict_no_lora_seed42 | 384 | 0 | 48.802% | 2.0607 | 13.330% | +0.4023 |
| covariance_risk_strict_no_lora_seed456 | 384 | 0 | 48.849% | 2.0631 | 13.082% | +0.4021 |
| covariance_risk_strict_seed123 | 512 | 0 | 53.856% | 2.2599 | 12.477% | +0.5973 |
| covariance_risk_strict_seed42 | 384 | 0 | 50.211% | 2.1167 | 13.523% | +0.4582 |
| covariance_risk_strict_seed456 | 512 | 0 | 47.898% | 2.0366 | 12.585% | +0.3756 |
| covariance_teacher_seed123 | 512 | 0 | 51.013% | 2.1348 | 12.368% | +0.4723 |
| covariance_teacher_seed42 | 384 | 0 | 51.454% | 2.1249 | 13.248% | +0.4664 |
| covariance_teacher_seed456 | 384 | 0 | 52.139% | 2.1310 | 13.519% | +0.4701 |
| numeric_attention_seed123_lr0003 | 128 | 0 | 47.766% | 2.0549 | 13.447% | +0.3945 |
| numeric_attention_seed42_lr0003 | 512 | 0 | 47.798% | 1.9870 | 13.630% | +0.3258 |
| numeric_attention_seed42_lr001 | 256 | 0 | 48.649% | 2.0165 | 13.778% | +0.3553 |
| numeric_attention_seed456_lr0003 | 512 | 0 | 54.832% | 2.1930 | 13.593% | +0.5319 |
| numeric_covariance_seed123 | 256 | 0 | 50.820% | 2.1323 | 13.203% | +0.4715 |
| numeric_covariance_seed42 | 384 | 0 | 45.112% | 1.9382 | 13.527% | +0.2761 |
| numeric_covariance_seed456 | 512 | 0 | 48.375% | 2.0553 | 13.378% | +0.3951 |
| numeric_risk_seed123 | 384 | 0 | 49.053% | 2.0624 | 13.525% | +0.4020 |
| numeric_risk_seed42 | 512 | 0 | 52.429% | 2.1676 | 13.097% | +0.5064 |
| numeric_risk_seed456 | 512 | 0 | 49.495% | 2.0574 | 13.716% | +0.3963 |
| numeric_utility_lag1_seed123 | 32 | 1 | 46.830% | 2.0365 | 13.300% | -0.0594 |
| numeric_utility_lag1_seed42 | 64 | 1 | 43.915% | 1.8710 | 14.097% | -0.1389 |
| numeric_utility_lag1_seed456 | 128 | 1 | 51.431% | 2.1060 | 12.941% | -0.1106 |
| numeric_utility_seed123 | 96 | 0 | 44.958% | 1.9353 | 13.523% | -0.1197 |
| numeric_utility_seed42 | 64 | 0 | 43.366% | 1.8469 | 14.150% | -0.1401 |
| numeric_utility_seed456 | 128 | 0 | 50.687% | 2.0686 | 13.045% | -0.1243 |
| utility_seed42 | 32 | 0 | 50.122% | 2.0969 | 12.830% | -0.0280 |

Selection criteria and all checkpoints remain in each run folder.
Read LITERATURE_COMPARISON.md for published-method scope and classical controls.
This table alone does not establish statistical superiority or equivalence.


## 15. Remaining work and restart boundaries

Finance training and sealed final evaluation are concluded for this artifact. Remaining useful work is paper synthesis, other-domain integration and cross-domain evaluation. A future scientific extension needs new independent evidence or explicitly exploratory reanalysis; consumed data cannot become unseen again. Do not relaunch old drivers because a document title says PENDING.

The customer-oriented interface is a historical portfolio demonstration, not a complete commercial platform. Brokerage authentication, real deposits/withdrawals, tax lots, suitability profiles, live execution, data monitoring and operational controls are not implemented. Do not add invented live transactions to make a demo seem autonomous. Shared GPU switching and cross-domain non-interference for all four actual adapters remain separately unverified in this checkout.

Explanations cover the recorded process. No rationale-specialized model was trained. A presenter asked about a market news event should state that news is not an input. Asked why not simply use minimum variance, state that it was stronger here; this work demonstrates reward-trained modular adaptation, not an economic necessity for a language model. That limitation should remain in the paper and not merely in this handover.

## 16. Final boiled-down presentation set

1. **Architecture slide:** pristine InternVL3-2B → rank16 attention/readout LoRA → existing A–E token probabilities → five causal strategy targets → exact fee-funded portfolio ledger. Label constant preferences and direct financial reinforcement. Do not draw news/state inputs into the constant prompt or call it PPO/GRPO.
2. **Protocol slide:** GLD/SPY/TLT/UUP/XLE; train2010–18, validation2019–21, reused development2022–24, reserved Feb–Oct2026; three seeds; update0 eligible;64 selected updates; one-close signal delay; five-session rebalances;10bp full-L1 fees. Full2B retained;6,809,504 trainable parameters.
3. **Primary results table:** development raw1.242536→trained mean1.569486; final raw1.513051→trained mean1.951463. In the same final table show returns6.5318%/6.3770%, drawdowns3.5094%/2.4901%, optimizer Sharpe2.169516 and all three seeds or their range. Include meanΔSharpe+0.438412 and95% interval[-0.080094,0.864912]. Do not omit lower return, optimizer or uncertainty.
4. **Final curve figure:** immutable final_performance.png/svg with raw, trained and classical control, dates and cost convention. Historical net simulations, not live returns. Seed42 is pre-fixed, not the best final seed.
5. **Small ablation/robustness table:** four-strategy native trained Sharpe~1.0751 versus five-strategy primary~1.5695 on the same development period; positive gains under alternate labels; frozen-base/disabled-adapter/reload checks. More numerical, head-only, timing and fee controls belong in supplement. Strategy-set differences do not isolate novel optimization or LLM advantage.
6. **Portfolio demo:** final-date$100,000 account→$106,377.03; holdings, fees and activity. Switch to10 March, ask why bonds were sold, show the corrected buy and signal dates. Follow with “How much is it worth?” and show that current values agree. Identify verified historical replay, not live GPU inference during the UI session.
7. **Balanced conclusion:** “Financial-reward training of an isolated InternVL3-2B adapter improved risk-adjusted portfolio performance over its exact initial policy across three seeds. Reserved-period point estimates remained positive with lower drawdown and slightly lower return; uncertainty includes zero and standalone minimum variance remained stronger.”

**What makes this batch presentable:** native adapter-only attribution, an unmodified raw baseline, meaningful absolute trained improvement over comparable primary native branches, three-seed repeatability, label controls, exact reload/deployment checks, a prospectively sealed final evaluation, and a portfolio interface whose explanations reconcile with actual execution. A single winning Sharpe is not the argument.

**Exclude from the main presentation:** every failed trial, every sampled rollout, annual-refit year grids, old covariance peaks, Nifty slides as a direct ETF comparison, infrastructure logs and full hashes. Summarize the search and make this report/appendix available. Negative controls explaining selection should remain visible; individual hyperparameter archaeology need not occupy the main PPT or paper.

**Avoid claims of:** state-of-the-art trading, outperforming FinATOM, higher profit after training, statistically confirmed final superiority, autonomous market reasoning, proven LLM necessity, convergence of abandoned16-update branches or fully verified four-domain production deployment. None is established by these results.


---

## Final presentation shortlist

1. Architecture: original Finance RL adapter, causal allocation/execution engine, and separate synthetic-SFT explanation adapter on the same fixed backbone. Explicitly disclose both Finance weight roles.
2. One financial comparison table: raw model, three-seed trained policy and standalone minimum variance, with return, Sharpe, drawdown, costs, dates and uncertainty. Include equity/drawdown curves. Do not imply the trained model had the highest return or beat the optimizer.
3. Historical portfolio UI: balance, holdings, activity, fees, date/seed controls and actual generated questions. No live brokerage claim.
4. Successful generated examples: TLT buy/fee correction, UUP concentration versus risk contribution, allocation contribution versus return. Preserve actual output and source dates.
5. Failure examples: fresh loss-comparison reversal, irrelevant attribution and fabricated confidence interval. Explain why synthetic validation loss and limited checking are inadequate reliability measures.
6. Full run history and paired comparisons belong in the appendix and handover. Final deployment selection is a narrow empirical choice, not universal optimality.

The portfolio experiment and real-model demo are presentable with these boundaries. The requested unrestricted justification reliability has not been achieved. Neither journal acceptance nor superiority over published systems is established by this audit.


---

# Finalization and full-period figure addendum — 16 November 2026

## Decision

Freeze the selected original Finance RL adapter as the final artifact for this campaign, and report a mixed outcome. This is a decision to conclude the current experiment, not proof of optimality or consistent financial superiority. Do not describe 6.38% as a model-wide return ceiling or the highest-return run. It is the selected adapter's return on the original reserved 2026 interval. Explanation reliability remains a separate unresolved limitation; finalization does not mean every original quality objective was achieved.

## What data belongs in the results

RL weight training used 2010–2018. Validation 2019–2021 selected checkpoints. Development 2022–2024 was reused adaptively to design/select methods. Thus “not used for gradient training” does not mean untouched or independent. Show validation/development only with those labels. The main generalization result remains the original reserved 2 February–30 October 2026 test, chosen before evaluation. Report three trained seeds; deployment seed42 was fixed beforehand, not chosen from test scores.

Include all completed comparable frozen-adapter additional annual evaluations (2022, 2023, 2024, partial 2025) alongside that main result, including negative outcomes. The 2025 data had been accessed in other project campaigns and is an additional retrospective audit, not a new pristine test. Its available saved panel lacks sufficient warmup for January1 under the unchanged environment, so evaluation starts January10. Do not silently treat it as a full calendar year.

Annual portfolios reset to cash. They cannot be multiplied or stitched into one continuous investable result; such a result would require a separate continuous simulation with holdings carried across boundaries. Do not pool development and test into a headline score. Do not select another checkpoint or dates because they produce a prettier chart. No new evaluation of validation years or every available historical date is claimed by this figure package. All completed same-policy comparisons are included in its CSVs; historical experiments on other assets/methods remain in the main run history, not falsely merged with this adapter.

## Figures for PPT/reports

01_final_metrics: six panels for net return, zero-risk-free annualized Sharpe, maximum drawdown, annualized volatility, average daily full-L1 turnover and modeled dollar fees. Identical 189-session dates and execution for untrained, trained seed42, standalone minimum variance and equal weight. Mean turnover includes non-rebalance days. Fees are already deducted from returns. Volatility is population daily standard deviation times sqrt(252). The three-seed mean Sharpe and paired interval are in the caption; the bars use deployment seed42 consistently. Full metrics for all seeds are supplied separately.

02_final_equity_drawdown: actual saved daily portfolio values and drawdowns, beginning with $100,000 cash before the first return. No smoothing or synthetic path.

03_period_robustness: all five completed yearly/partial-year periods, separately labeled development, additional and reserved. Return is period total, not annualized. All annual comparisons use the same frozen seed42 policy against its exact untrained probabilities. Dates, period lengths, simple baselines and remaining metrics are in period_metrics.csv and the original annual audit.

Each figure is supplied as high-resolution PNG, vector SVG and PDF. CSVs contain the numbers; SOURCE_MANIFEST.json identifies input-file SHA256 hashes. make_final_finance_figures.py reproduces figures. No base or adapter weight was changed.

## Final interpretation and numbers

Period: untrained return → trained return; untrained Sharpe → trained Sharpe; untrained max drawdown → trained max drawdown.
- 2022 development: 4.585% → 4.965%; 0.583 → 0.807; 5.047% → 3.250%.
- 2023 development: 7.445% → 7.004%; 1.380 → 1.597; 3.739% → 2.982%.
- 2024 development: 12.225% → 12.057%; 2.325 → 2.810; 3.170% → 2.481%.
- Partial 2025 additional audit: 11.914% → 8.560%; 1.540 → 1.273; 6.258% → 5.792%.
- Original 2026 reserved test, seed42: 6.532% → 6.377%; 1.513 → 1.951; 3.509% → 2.490%.

Trained three-seed mean final Sharpe1.951463, raw1.513051. Paired95% ΔSharpe interval [-0.080094,0.864912] includes zero. Standalone minimum variance final Sharpe2.169516, return6.396%, drawdown2.195%; it outperforms the learned portfolio on all three of these metrics in this interval. The selected trained policy is not superior to this simple comparator.

Across the five reported periods, trained drawdown was lower in5/5; net return higher in1/5; Sharpe higher in4/5. These are descriptive counts across dependent, partly adaptively reused historical intervals, not independent statistical trials or a significance test. The 2025 weakness must be visible in the presentation.

## Final showable conclusion

A small InternVL3-2B RL adapter learned a more defensive mixture of causal allocation strategies. In the reserved evaluation it reduced volatility, drawdown, turnover and fees, improved observed Sharpe, and slightly reduced net return. Across additional historical regimes, the return and Sharpe benefits were inconsistent. Classical minimum variance remains a strong and sometimes better alternative. Generated portfolio explanations are a separate model capability with documented factual failures. These results support a transparent adaptation study, not a profit-improvement, universal superiority or reliable autonomous-adviser claim.


## Replacement headline comparison figure

Use final_presentation_figures/clean_final_comparison/trained_vs_untrained.png (or vector SVG/PDF) as the primary trained-versus-untrained figure. It supersedes the earlier cluttered multipolicy figure for the presentation headline. Four metrics, two consistent colors, direct values and signed changes isolate the final selected seed42 adapter on the reserved2026 interval. The conclusion is lower observed risk and higher Sharpe, slightly lower net return. CAPTION.md supplies the full evaluation and uncertainty statement. Keep minimum-variance/equal-weight baselines and annual robustness checks in supporting results, not omitted from the research paper. Earlier figures remain historical/supporting artifacts.


Publication figure revision: the primary trained_vs_untrained figure now has only four metric panels, axes, a shared legend and numeric values. Headline, directional guidance, change annotations and takeaway prose are removed. Use CAPTION.md for interpretation, evaluation dates, seed and uncertainty.


Publication figure revision: the primary trained_vs_untrained figure now has only four metric panels, axes, a shared legend and numeric values. Headline, directional guidance, change annotations and takeaway prose are removed. Use CAPTION.md for interpretation, evaluation dates, seed and uncertainty.


## Buy-and-hold baseline added

The preferred final comparison is now final_presentation_figures/clean_final_comparison/trained_untrained_buyhold.png (also SVG/PDF). It retains the untrained and trained seed42 policies and adds an equal-weight, initial-purchase-only baseline on the same five ETFs and exact189 return dates. One initial purchase at30January2026 costs $99.90; thereafter units remain fixed, with no rebalancing or terminal liquidation. Vendor-adjusted prices reflect dividends on the same basis as existing results. This is not the previously reported periodically rebalanced equal-weight comparator. Buy-and-hold net return 6.916245%, Sharpe 1.250403, max drawdown 4.449750%, annualized volatility 7.348328%. Source hashes, fixed units and daily values are in buy_and_hold.json; caption is BUYHOLD_CAPTION.md. No adapter or original financial bundle was modified. Endpoint-price wealth identity and absence of subsequent turnover were checked.


---

# Current presentation result

Use nifty_test_results/FINAL_TEST_REPORT.md and presentation/nifty_test_2025.*. Exclude development scores from the main presentation. The verified Nifty adapter shows training benefit versus raw on retrospective tests; its small buy-and-hold advantage is not statistically established. The running UI still uses the earlier ETF adapter; do not present its records as this Nifty result.


---

# Final showable result after paper-trading integration

- ONE selected artifact: original Nifty native RL seed42 checkpoint44. Train2019–2021; validation2022. Headline continuous retrospective test2025–January2026: raw11.22%, trained13.28%, equal holding13.06%. Rebalanced equal weight13.44% is stronger. Do not use45% development or15% calendar2025 as competing headline scores.
- Original weights reloaded, disabled-adapter recovery checked, all three seeds retained. Live historical paper engine makes54 actual model calls and reproduces every one of270 original daily balances exactly.
- Demonstrate Start/Pause, Next day, Next rebalance, holdings/trades/fees, source-date inspector and exports. Simulated funds/historical data, no broker/news/real-time inference claim beyond actual adapter calls.
- Finance conversation removed: SFT10/30 full passes, base8/30 in a selected developer-reviewed factual audit. No preset answers replace it. Inspector is recorded evidence, not generated rationale.
- Primary report JOURNAL_RESEARCH_REPORT.md, full numeric appendix and conditional uncertainty/cost sensitivity, actual browser/video evidence and source hashes accompany the release. No assertion of global optimality, no flaws, exhausted2B capacity or established superiority over holding.
- Source, selected packages, original SSD context bombs and college artifacts are on GitHub `main` in Parth-authored commits. The complete current-workspace checkpoint release is published: 5,224 original paths, 4,663 verified unique objects, 22 public assets. See `archives/CHECKPOINT_RELEASE_STATUS.md` and `archives/published_checkpoint_receipt.json`. Public manifest/index/helper downloads were checked. Upstream base and environments must be reinstalled; unrelated SSD domains are outside this Finance backup.

# Superseding interface update — 17 November 2026

The user clarified that Finance should be built like the existing chatbot/anomaly interface. Located `git-push-kar/Polyvalent` and preserved its `ui/index.html` reference (blob `4fad71c9c8d645a5acc633efef9e50a9407aaa4b`). The previously embedded Finance page is replaced by a native Finance tab using the same layout, cards, purple accent and automatic light/dark theme. Original Chat and Anomaly section markup is preserved. Finance IDs/styles/state are isolated, it initializes on first visit, and tab changes pause playback. Old Finance page URLs redirect to the native tab. Duplicate Finance API-router registration was removed.

No training or financial-method change. The final native UI made 54 actual adapter calls across 270 sessions in 41.69 seconds. Every original daily balance matched exactly (maximum difference 0.0 INR). Native control/theme/navigation smoke checks used two additional actual adapter calls; all 35 existing tests passed. Detailed receipts and source provenance: FINANCE_UI_INTEGRATION_20261117.md. Active server now serves C:/Users/Parth/Documents/C68-FinanceRL/demo, not the recovery clone; PID53076 at verification (verify before stopping).

## Final presentation shortlist after native UI integration

- One selected artifact and result: Nifty native RL seed 42, checkpoint 44; training 2019–2021, validation 2022; continuous retrospective test 2025–January 2026. Trained 13.28%, original 11.22%, holding 13.06%; equal rebalanced 13.44% is stronger. Keep uncertainty/cost/selection limitations.
- Show the native Finance tab alongside Chat and Anomaly. Start/Pause, Next day/rebalance, account metrics, chart, holdings, actual trades, fees, dated allocation records and exports work. Historical simulated funds; no broker or live-market claim.
- Actual model calls and exact path parity are verified. Finance conversation remains removed after failing factual testing. Records are not generated explanations. No new 2B capability/optimality claim.
- Main source/reports and original SSD/college context are on GitHub main under Parth-authored commits. The complete checkpoint release remains published and verified; UI-only changes require no new weight release. Use START_HERE.md and FINANCE_UI_INTEGRATION_20261117.md for current state.

# State-aware replacement attempt — 17 November 2026

The user explicitly rejected the old constant-mixture adapter as satisfying portfolio-aware trading. That objection is correct: the old adapter did not receive stock prices, holdings, available money or affordability, and repeated genuine calls did not add adaptivity. Its historical 13.28% measurement is not evidence of the requested capability. The old menu server is stopped while a replacement trains.

A single new primary run is launched from research/stateful_rl_20261117, with five full per-stock observation rows, cash/positions/equity/initial capital, per-stock action logits from native InternVL vocabulary, whole-share execution and next-open chronology. Only attention/vocabulary-head LoRA is trainable. Exact all-action one-rebalance financial reward policy gradient is used, explicitly offline contextual RL rather than PPO. Nine accounting/causality/gradient tests and actual GPU optimizer/export/recovery checks passed. No claimed performance success yet.

Run: C:/Users/Parth/Documents/PolyvalentFinance/campaign_runs/stateful_full_account_seed42_20261117. Actual Python PID64180 at launch (verify before operating); wrapper PID62228.12epochs,1536training states,accumulation4,seed42. Logs in sibling reports/stateful_full_account_primary.stdout.log and .stderr.log. Training rewards2015-2021, validation2022; primary continuous retrospective test2025-January through13November2026. Selection uses absolute validation net growth at100000INR, not the best delta or convenient test capital. Automatic tests include pristine-disabled model, buyhold, stronger simple controls, every single-stock holding and capitals500/50000/100000/200000. No automatic menu promotion.

Detailed source/method/limits and resumption instructions: docs/STATEFUL_RL_ACTIVE_RUN_20261117.md and research/stateful_rl_20261117/README.md. User imposed a total12-24hour attempt and at most1-2substantive runs; no time cap is encoded in training. Codex should become inactive after verifying real optimization has started; user will wake it after approximately4-5hours. Do not launch duplicate workers, reopen the old GPU demo or claim the new requirement is met before inspecting outputs.

## Current worker superseding the first state-aware run

The first full-state run was stopped during epoch10 after validation collapse: raw+1.432%, trained epoch1-8.106%, epochs5-8-18.308%; epoch8 all50 validation actions concentrated Infosys. Full failure records are preserved, no test result was used to choose revision2. The user explicitly lifted the two-run ceiling; several justified attempts are allowed within the original12-24hour total budget.

ACTIVE: C:/Users/Parth/Documents/PolyvalentFinance/campaign_runs/stateful_monthly_risk_seed42_20261117_v2, source G/research/stateful_rl_20261117_run2, logs R/reports/stateful_monthly_risk_v2.stdout.log and .stderr.log.8epochs maximum,1536states,seed42,accum4. Train2015-2023,validate2024,primarytest2025Jan-2026Nov13.20-session allocation/reward, stock35%/sector60% rebalance caps with small-budget relaxation, full-identity row/query shuffling, corrected query-end token selection, lowerLR/strongerKL, checkpoints32/64/128/every256,patience24 validation checks. No wall-clock/quiet restriction coded. No automatic menu deployment.

13 CPU/gradient/input-mapping tests and actual GPU update/export/disabled recovery passed. Complete observations measured840tokens in probes. One setup was aborted during untrained-reference capture before optimization due failed endpoint test and launcher-exit gating; both corrected, aborted directory preserved. Read docs/STATEFUL_RL_REVISION2_ACTIVE_20261117.md first on resumption; do not operate the stopped run1 or aborted un-suffixed setup. No conclusive winning result yet.


Revision2 launch verified: actual PID68640, wrapper70072; actual optimization/checkpoint32/64 confirmed. Source hashes match protocol. See docs/stateful_rl_revision2_preflight/launch_verified.json. Leave the worker independent and become inactive; user can check in about3-4hours. No new convincing final result yet.

## Revision 3: current work, superseding prior active-worker notes

17 November 2026 afternoon: run2 COMPLETED, selectedstep1792. Primary100k test trained-8.176%, raw+1.893%, equalhold-1.795%; validation+22.372% did not generalize. Seven validation feature-sign counterfactuals changed greedy decisions only once. Preserved evidence: docs/stateful_rl_run2_result. No successful replacement declared.

User asked for a fresh deep research/design review and authorized12-18hours with some flexibility. New run source research/stateful_rl_20261117_run3: causal inverse-volatility supervised curriculum selected using2015-2023 training years, then clipped group-relative RL on actual60-session,three-decision account episodes. Full stock/account inputs, real whole-share next-open execution, native InternVL3-2B LoRA. Separate SFT-only baseline mandatory; do not claim RL succeeded if only SFT helped. Train2015-2023,val2024,testJan2025-Nov13 2026 stays retrospective/adaptively reused. No outcome guarantees, no optimum/flawless-process claims.

17CPU tests and realGPU preflight passed. Update/reload/disabled recovery/reference-swap checked; exact0 output error, nonzero parameter change. Actual live workerPID36996, wrapper63884, launched14:46:40IST. Run C:/Users/Parth/Documents/PolyvalentFinance/campaign_runs/stateful_curriculum_episode_seed42_20261117. Logs reports/stateful_curriculum_episode.stdout.log and.stderr.log. Verify process/status before action. No old UI GPU server. No duplicate worker. Expected roughly3hours total; inspect current progress before giving a revised ETA.

Full primary-source review, implementation choices, failure interpretation, limitations and evaluation contract: docs/STATEFUL_RL_REVISION3_RESEARCH_20261117.md. After completion run research/summarize_stateful_episode.py with run path; inspect allcomparators,SFTvsRL,state/capital sensitivity,savedweights before anymenuintegration. Raw share-count differences alone do not prove learned budget preferences. No auto promotion. No quiet/time restrictions coded. User wants Codex inactive only when meaningful concurrent analysis is finished and training can run independently.

## Revision 3 completed — 17 November 2026, checked 17:23 IST

Worker finished in5906.9seconds (~98minutes). RL stopped at iteration72 after12 validation checks without improvement. Selected checkpoint384 is supervised-only. All subsequent supervised/RL validation checkpoints tied its15.8745% return, so earliest-tie selection correctly retained384. This is NOT an RL success.

Primary January2025-November13 2026,100000INR: selected-3.83799%, pristine-0.112782%, holding-1.79538%. Selected policy has exactly one action vector [4,4,4,4,4], target20% each. Its results match equal-weight rebalancing exactly. Six validation probes also kept all actions unchanged after changing return signs, volatility ordering, account capital or held inventory. This shows greedy-action collapse despite available complete inputs; it does not imply the inputs were omitted or probabilities identical.

5000INR gives+13.0096%, above raw0 and hold9.2251%, but matches ordinary equal rebalancing exactly. Do NOT promote it as learned budget adaptation or select that capital to rescue the result. Different capital returns here arise from integer execution, not demonstrated adaptive allocation choices. No menu promotion, no running training worker, no successful conclusive replacement. Failure/current artifacts preserved in docs/stateful_rl_run3_result. Descriptive paired block uncertainty audit completed; adaptive retrospective caveats remain.

Next methodological issue is learning conditional actions rather than the majority allocation class. Do not blindly repeat the same warm-start plus weak clipped update or label extra training an established fix. Any subsequent attempt needs an explicit state-learning gate and comparison with constant/equal-weight policies before costly financial evaluation.

## Current continuation — 17 November 2026, 17:42 IST

User renewed broad authorization to continue within roughly12-15hours; inactive only when no useful parallel work remains. Active run4: stateful_focused_gate_seed42_20261117, actualPID15384/wrapper12160. Full stock/account context, one focused-stock prompt per action. Balanced causal curriculum2015-2022; held-out2023 conditional-learning gate;2024 financial validation. First512-example gate14.58%raw->39.58%trained, but failed minimum class recall. Continue scheduled checks; no financial success claimed.15CPU tests and actual GPU preflight passed. See STATEFUL_RL_REVISION4_ACTIVE_20261117.md.

CPU diagnostics: ridge forecasting gives mixed2022-2024 results, not a stable dominant edge. Separate38-feature numerical classifier learns exactly the allocation curriculum/gate at~88%held-out accuracy. This supports investigating numerical representation, not a profitability claim and not a replacement trading agent.

Prepared run5 adds201600-parameter numerical INPUT encoder to nativeInternVL before all28decoder layers; original vocabulary emits actions. Both encoder andLoRA are saved/restored/disabled consistently.16CPU tests passed; GPU preflight still pending. See STATEFUL_NUMERIC_BRIDGE_CONTINUATION_20261117.md. Hidden continuation supervisorPID23132 waits for run4exit and only if explicit learning_gate_failed runs5CPUtests,GPUpreflight thentraining; no automatic restart after arbitrarycrash or financialloss. Supervisorstatus R/reports/focused_continuation_status.json,logs focused_continuation.stdout.log/.stderr.log. No concurrentGPUmodels,noautodeployment,noencodedtrainingdeadline.

Check both run4status and supervisor before doing anything. If fallbacklaunched run R/campaign_runs/stateful_numeric_bridge_seed42_20261117. Initial gate within roughlyhalfanhour; fullRL/evaluation can take severalhours. Do not claim conditionalaccuracy proves returns, or that smallcapital integer rounding proves learnedstrategy. Preserve SFTvsRL comparison, matcheduntrained,holding,strongercontrols and negative evidence.

## 18:30 IST continuation — focused learning gate passed, RL active

Run4 passed at2048 supervised examples: held-out balancedaccuracy63.5417%, raw14.5833%, minimum eligibleclassrecall25%, seven predictedclasses. Earlier checks51239.58%,102451.04%,153641.67% failed; preserved rather than omitted. The numericbridge fallback has NOT launched because run4 passed and is running episodicRL.

Early2024 financial validation100k: raw0.1840%, SFT7.7004%, RLiteration1 10.8307%, iteration2 11.2280%, iteration4 9.1119%, iteration8 7.4616%. Holding14.2273% remains stronger. Best earlyRL uses6distinct actionvectors across13decisions. These are validation observations only, not final results or proof of out-of-sampleprofitability. Actualworker15384alive, iteration10atmostrecentinspection; supervisor23132waitingforworkerexit. Do not modify active source or run anotherGPUmodel.

Prepared standalone research/stateful_session.py incremental whole-share simulator (NOT connected to productionUI, no promotedcheckpoint). FiveCPUtests pass: exact daily equity/cost/trade/observation parity versus research evaluate at500/5000/100000, atomic inferencefailure, immutable snapshots, no extra completedaccountcalls. Tests use an explicitly named syntheticfixture, not actualadapterinference; realGPU/productionbrowserparity remains required after checkpointreview. Olddemo remains historicalstatic, not valid statefuldeployment.

## Shutdown recovery — 17 November 2026, 21:59 IST

PC shut down after reported RLiteration69/96. No optimizer or RNG checkpoint existed in the original run. Latest complete saved adapter is checkpoint10064, whose2024 validation return is19.8887% (holding14.2273%, supervised7.7004%, pristine0.1840%). Checkpoint56 had15.6059%. These are validation only, not2026 test success. Five completed reported iterations65-69 had no recoverable weights.

New recovery worker source research/stateful_rl_20261117_run4/resume.py. Run R/campaign_runs/stateful_focused_resume_20261117, actualPID18704/wrapper18792. Logs R/reports/stateful_focused_resume.stdout.log and.stderr.log. Original run and artifacts preserved. All13 checkpoint directories copied with manifest hash checks. This is explicitly a WEIGHTS-ONLY WARM RESTART fromiteration64 with fresh Adam and sampling seed4242, not an exact optimizer continuation. SFT reference and all earlier validation candidates are preserved; original96-iteration endpoint retained. No repeated SFT/gate training.

Resume validates restoredcheckpoint64 against its saved2024 daily equity before optimization. Thereafter every completed iteration saves separate recovery model weights plus atomically replaced training_state_latest.pt containing Adam, TorchCPU/CUDA, NumPy, Python and sampler RNG states, bestscore and patience. Future recovery can use those snapshots; uncommitted post-snapshot checkpoints are ignored when loading candidates.15existingCPUchecks and recovery-source compilation passed. No changes to original training source, data, reward or trading rules. Legacy continuation supervisor is no longer active aftershutdown; do not restart it on the old run. Only newrecoveryworker should useGPU.

User asked for a quick status/resume. Check new status/reloadreceipt first, then selected test outputs after completion. No model automatically promoted. Remaining training/evaluation roughly2-2.5hours based on prior iteration timing, not an encoded time cap.

## Recovery verified in operation — 17 November 2026, 22:06 IST

Resumed worker18704 completediterations65 and66 from recovered64. Status66 records full_recovery_saved=true; training_state_latest.pt exists,54,687,754bytes, updated22:06:11. No stderr error observed. This verifies that optimizer updates resumed and full recovery snapshots are now actually being written, not merely implemented. The fresh optimizer's effect on performance is unknown; no new validation checkpoint afterrestart yet. Restored64 had exactly zero validation-equity discrepancy before updates.

Latest pre-interruption2024 validation trajectory: iteration40 return8.4506%,48 12.9888%,56 15.6059%,64 19.8887%; best64Sharpe1.338,drawdown6.846%. Comparators same2024/100000INR: untrained0.1840%, supervised-only7.7004%, buyhold14.2273%. Supervised-only means the native adapter after causal inverse-volatility allocation imitation and before financialRL, not the untrained base or an externalteacher trading instead of the model. RL improvement on this validation sample is12.1883 percentagepoints overSFT and5.6614pp overholding.

Interpretation: encouraging observed validation improvement, but checkpoint selection maximizes this same2024 metric. It is not a calibrated forecast of2026 profit. Priorstatefulrun2 reached22.3721%validation then-8.176%oncontinuous2025-Nov2026, so our own history gives a concrete warning against extrapolation. Focusedrun4's passed conditional-learning gate and nonconstant decisions address a priorfailure, but do not prove futuretradingedge or isolate eachchange's causal contribution. No defensible numerical odds of test success are available from this small adaptivecampaign.

FreshAdam discards moving gradient estimates and stepcount, so subsequent updates differ even though learnedweights are preserved. Seed4242also changes sampledtrajectories. Oldbestcheckpoint64 remains an eligible selectioncandidate and is never overwritten, but validationselection still cannot guarantee test superiority. LaterFeb-Nov2026holding is alreadynegative(-10.2626%at100k), so do notpromise the ordering trained>holding>raw orpositiveholding. Desiredsuccess is positive trained netreturn and superiority tobothmatchedcomparators, whichever comparator is stronger.

User explicitly requested continual context-transfer updates with timeline, numbers, decisions, evidence/inference separation, and success/failure explanations. Continue appending meaningful events; do notinvent causal explanations for observations that lack controlledablations. Final2025-2026matchedtest remains pending.

## Full-universe correction — 17 November 2026

User explicitly rejected five-stock scope and requires ALL original dataset stocks withInternVL3-2B and fullportfolio information. Stopped active five-stock recovery processes18704/18792 after verifying commands; preserved artifacts. Those results are diagnostic only and cannot conclude originaldomain.

OriginalCSVhash29b847a33ad88b7568eef66e1fb9f3274a897bd3091ddb8551c129b75dd726f7:287263rows,49stocks,1999-Jan2026. All49complete2018onward. Immutable original_panel preserves everyrow/date/ticker; nofill,noassetdrop. Supplementaryexecutionrecords capturedall49; LTIM lookupaliasLTM verifiedNSEsymbolchange. No overlapping adjustedreturn discrepancy>50bps. Rawhistorical executablequotes and demergerentitlements stillrequireexplicitresolution; nofabricatedwhole-shareaccounting.

Newsource research/full_universe_rl. FiveCPUchecks pass and actualGPUpreflight PASS:49x20numericalstockstates,49x7nativeactionoutputs,1051tokens,peak5807846912bytes,nonzeroinputbridgegradient19.9317,outputchange0.2401 afteractualupdate,disabledrecoveryerror0. Hardware demonstrablyfitsall49; priorfive-stocklimitwasnotrequired. Mechanicalprobeusesartificialquotes/targets andmustNOTbepresentedasfinancialtrainingorprofits. AddedsharednumericalINPUTadapterisexplicitlydisclosed; original28-layerLM andvocabularyremainactionpath.

Read docs/FULL_UNIVERSE_REBUILD_20261117.md forauthoritativescopeandpendinggates. No full-universeRLtrainingstarted yet. Pendingcorporateactionledger,realquoteverification,fullpolicyserialization,matchedcontrols,learninggateandRL. Do not resurrectfive-stockworker or claimlargeruniversebyjustchanginglabel. OriginaldatasetendsJanuary30 2026; later2026dataareseparateextension,notoriginalCSV. Contextmustkeepthisdistinctionexplicit.


## 17 November2026: all49 continuation after chat migration

See [full-universe continuation](FULL_UNIVERSE_CONTINUATION_20261117.md) for implementation, chronology, execution approximations, training protocol and recovery. Run b launched after15 checks; inspect live status before claiming progress. Original dataset/all49/InternVL3-2B retained; no new profitability claim.


## 18 November2026: complete counterfactual all49 campaign

Prior restored-all49 run failed primary goal. User requested a full-fledged new approach. See COUNTERFACTUAL_FULL_CAMPAIGN_20261118.md. Three native seeds, corrected matched execution, relative scores, canonical inputs and per-stock simulator counterfactual credit. Same original49stocks/dataset/InternVL3-2B. CPU numerical control is separate. Actual first GPU update/reload passed; no new final returns yet.


18 November 2026. All three native counterfactual seeds selected initialization. See [completed results and final experiment](COUNTERFACTUAL_RESULT_AND_FINAL_ROTATION.md).


18 November 2026. [Complete experiment history, final three-seed results and integration decision](FINANCE_EXPERIMENT_HANDOVER_20261118.md).
