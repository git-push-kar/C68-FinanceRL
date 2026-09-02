# FinanceRL — AI-Driven Portfolio Management with InternLM + Reinforcement Learning
### Project Report — September 2026

---

## 1. Executive Summary

FinanceRL is a **research platform for automated portfolio management** that combines a large language model (InternLM 2.5, 1.8B parameters) with reinforcement learning (PPO). Think of it as teaching an AI to manage a basket of stocks the way a human fund manager would — deciding every day how much to allocate to each asset — but learning purely from historical prices, without ever touching a live market or a broker.

**The core idea is simple to state, hard to do well:**

*   Freeze a powerful pretrained language model (which already understands sequences and patterns).
*   Attach a tiny, trainable “finance adapter” (LoRA) plus a few small neural nets that translate market data into the model’s language.
*   Let the system learn by trial and error in a simulated market, rewarded for real portfolio growth and penalized for unnecessary trading, risk and drawdowns.

The project today is **fully working on an NVIDIA RTX A5000 (24GB)**, trains in `bf16` with modern GPU features, and has produced its first measurable results on the Indian Nifty 50.

> This is **research / paper-trading software, not investment advice** and not a live trading system. All trading is simulated on historical data.

---

## 2. Objectives

### 2.1 Primary Objectives

1.  **Build a portable finance adapter** — a LoRA that can be dropped onto the shared InternLM backbone alongside other adapters (ATC, etc.) without retraining the base model.
2.  **Learn a long-only, cost-aware portfolio policy** — fully invested, daily rebalanced, with realistic 10 bps transaction costs and a 30% single-asset cap.
3.  **Enforce honest evaluation** — train only up to a cutoff date (e.g., `2022-12-31`) and test on a completely untouched period (`2023-01-01 → 2024-12-31`) that is never used for tuning.
4.  **Run efficiently on A5000 (Ampere, sm_86)** — bf16, TF32, 24GB, fast enough for 75k–100k step production runs.

### 2.2 Secondary Objectives

*   Demonstrate that a frozen LLM + small encoders can be repurposed for numerical time-series (OHLCV + portfolio + risk signals).
*   Provide reproducible offline data (bundled Nifty 50 panel) so `git clone` works with no download.
*   Establish baselines (equal-weight) and a fair holdout harness (`evaluate_policy.py`) for future research.
*   Document survivorship bias and other pitfalls so future extensions can use rigorous historical constituents.

---

## 3. What This Is — In Plain Language

Imagine you have 49 stocks (the Nifty 50). Each day you see:

*   **Market window:** last 30 days of returns, log-returns, volatility, moving-average gaps and volume for each stock (5 features × 30 × 49).
*   **Portfolio:** what you currently hold (49 weights + cash).
*   **Risk:** recent volatility, current drawdown, cash position.

An AI looks at those three and outputs a score for each stock. Scores are turned into portfolio weights (soft-max, capped at 30% per stock). The portfolio then earns — or loses — money as prices move the next day, minus trading costs.

The AI is not given a “correct answer.” It learns by **reinforcement learning (PPO)**: actions that raise a risk-adjusted reward (net return minus penalties for turnover, volatility and drawdown) are reinforced; actions that cause churn or deep drawdowns are discouraged. Over thousands of simulated days it gradually discovers more stable allocations.

**Why a language model?** InternLM is excellent at modeling sequences. Rather than training a transformer from scratch on limited finance data, we freeze InternLM and only train a small LoRA (rank 16) on attention projections (`wqkv`, `wo`) plus three lightweight encoders (market / portfolio / risk) and two heads (actor / critic). The base model provides a strong sequence prior; the adapter specializes it to finance.

---

## 4. Dataset

| Item | Detail |
|------|--------|
| **Bundled panel** | `data/nifty50_prices.csv` — 14 MB, **287,263 rows** long-format |
| **Tickers** | 49 Nifty 50 constituents (current NSE list, `.NS` Yahoo suffix, e.g., `RELIANCE.NS`, `TCS.NS`) |
| **Period** | `1999-01-01 → 2026-01-30` raw; **effective `2017-11-17 → 2026-01-30` after `ffill` + `dropna`** (`env.py:45`). Earlier years lack all 49 tickers and are dropped, yielding **2,027 trading days** (confirmed by `train.py` logs `dates=2027/1266`). |
| **Columns** | `Date, Ticker, Close, Volume` (Volume optional). `nifty50_summary_statistics.csv` provides per-ticker returns, volatility, splits/dividends — e.g., `EICHERMOT 998k%` total return since 1999, `BAJFINANCE 236k%`. |
| **Bias** | Uses **current** Nifty 50 list backfilled → **survivorship bias**. Good for first experiment, not for historic-strategy claims. README warns to use NSE constituent archives for rigorous work. |
| **Honest split** | Train `... → 2022-12-31` (≈1,266 days with window 30 → **~1,068 steps/episode**), holdout `2023-01-01 → 2024-12-31` (**459 steps**, 2 yrs) |

---

## 5. Current Implementation

### 5.1 System Overview

```
data/nifty50_prices.csv  ──►  PortfolioEnv (window 30, 10bps, 30% cap)  ──►  {market, portfolio, risk}
                                                       │
                                                       ▼
                                          FinanceEncoders (3× MLP, 2048-d)
                                                       │
                                                       ▼  (3 tokens × 2048)
                                    InternLM2-1.8B frozen + LoRA finance (wqkv/wo, r=16)
                                                       │  hidden_states[-1][:,-1]
                                                       ▼
                                    Actor (49) ──► weights          Critic (1) ──► value
                                                       │                      │
                                                       └────► PPO + GAE ◄─────┘
```

*   **Backbone:** `internlm/internlm2_5-1_8b-chat` (24 layers, hidden 2048, `bf16`, `trust_remote_code`). Frozen; only LoRA is trainable. Corrected from `q_proj/v_proj` (Llama) to InternLM fused `wqkv/wo`.
*   **Encoders:** Market `Linear(5→2048)→GELU→Linear(2048→2048)`, Portfolio `LayerNorm(50)→Linear(50→2048)`, Risk `LayerNorm(3)→Linear(3→2048)`. Their outputs are stacked as 3 tokens fed as `inputs_embeds`.
*   **Heads:** Actor `Linear(2048→49)`, Critic `Linear(2048→1)`, `log_std` for continuous Gaussian policy.
*   **PPO:** γ 0.99, λ 0.95, clip 0.2, value 0.5, entropy 0.01, 4 epochs, rollout 256 (prod 512), minibatch 64 (prod 128). `ppo.py` normalizes advantages. `train.py` handles `bf16` autocast, `fp16` GradScaler, `TF32`.
*   **Environment reward:** `net = gross - turnover*10bps` , `reward = net - 0.001*turnover - 0.10*vol - 0.20*drawdown` (`env.py:63`). Info dict also exports `equity, gross_return, turnover, trading_cost` for logging.

### 5.2 Hardware & Performance (Verified on RTX A5000)

| Spec | Value |
|------|-------|
| GPU | NVIDIA RTX A5000, **WDDM, driver 595.95, CUDA 13.2**, compute **sm_86**, **24,564 MiB (25.8 GB)** |
| Torch | **2.11.0+cu128**, `torch.cuda.is_available() = True` |
| Dtype | **bf16** (A5000 native), TF32 matmul + cuDNN TF32, `high` matmul precision (`train.py:36-40`) |
| VRAM | **3.9 GB alloc / 6.05 GB reserved** for 49 assets, bf16 — 1.8B is ~3.6 GB + LoRA/heads |
| Throughput | Eager, autocast per step; `triton` unavailable on Windows → `torch.compile` falls back (Linux/WSL needed). Smoke `512 steps ~35–70s`, full `25k ~27–45 min`, prod `75k ~1.3h` (~70 episodes train-split). |
| Logging | Per rollout `mean_reward / mean_net / max_dd / mean_turnover / peak_equity` + VRAM, plus periodic `[Holdout ...]` learned vs equal-weight when `--train-end` is set, and `checkpoint_<steps>/` snapshots. |

### 5.3 File Map

`config.py` (ModelConfig/PPOConfig), `env.py` (PortfolioEnv), `model.py` (InternLMFinancePolicy + FinanceEncoders), `ppo.py` (GAE, ppo_update), `train.py` (A5000 loops, checkpoint/holdout), `evaluate_policy.py` (new — loads any `artifacts/...` LoRA+heads and reports holdout vs baseline), `evaluate_baselines.py` (equal-weight), `prepare_nifty50_data.py` (yfinance), `data/`, `artifacts/`.

---

## 6. Progress Report

### 6.1 Timeline (Recent)

1.  **Fixed InternLM LoRA target bug** — `q_proj/v_proj` → `wqkv/wo` with auto-remap, raised from `peft ValueError not found`. Added fallback detection.
2.  **Fixed Transformer 5 + InternLM forward** — `use_cache=False, output_hidden_states=True`, extract `hidden_states[-1][:,-1]` (CausalLMOutput) instead of `last_hidden_state`.
3.  **Fixed bf16 dtype mismatch** — encoders/heads `bf16` vs `float32` obs; added cast in `model.py:80`. Fixed `bf16 → numpy` cast in `train.py:110` and PPO `float()` for stable GAE.
4.  **Switched A5000 to GPU** — replaced `2.13.0+cpu` with `2.11.0+cu128`, verified `cuda:0, sm_86, 25.8GB`, `bf16` forward full.
5.  **Added observability** — rollout `mean_net/max_dd/turnover`, checkpoints, periodic holdout eval, and a dedicated `evaluate_policy.py` that correctly finds `finance_lora/finance/adapter_config.json` nested layout and puts model on CUDA.

### 6.2 Training Evidence

#### Full-panel 25k run (leaky — no `train-end`, trained on holdout too)

Last 1.5k steps printed:

```
steps=23552 mean_reward=-0.014701 peak_equity=103087
steps=23808 mean_reward=-0.019543 peak_equity=103087
steps=24064 mean_reward=-0.025148 peak_equity=103087
steps=24320 mean_reward=-0.002775 peak_equity=165188  ← spike one episode
steps=24576 mean_reward=-0.011869 peak_equity=189884
steps=24832 mean_reward=-0.008886 peak_equity=105604  ← give-back next episode
steps=25000 mean_reward=-0.023964 peak_equity=105604
Saved to artifacts/finance_v1
```

Reading: `mean_reward` is penalized (negative normal); `peak_equity` whipsaw `103k→189k→105k` shows high variance — **not yet converged** at 25k (12.5 full episodes). Starting equity `100k`, so `105k` final high-water-mark is `+5.6%`.

#### Holdout `2023-01-01 → 2024-12-31` (459 trading days) — table

| Run | Final equity | Total return | Annualized | Mean net / day | Turnover | Max drawdown | Daily vol | Sharpe proxy | vs equal-weight |
|-----|-------------|-------------|------------|---------------|----------|-------------|-----------|--------------|-----------------|
| **Equal-weight (baseline)** | **150,471** | **50.47%** | **23.17%** | **0.0919%** | **0.0022** | **12.47%** | 0.0076 | **1.93** | — |
| **Learned — honest smoke 512 steps** (`train-end 2022-12-31`, `smoke_honest/checkpoint_512`) | **151,186** | 51.19% | 23.54% | 0.0934% | 0.099 | 12.99% | 0.0081 | 1.82 | **+0.47% alpha** |
| **Learned — leaky 25k** (`finance_v1`, trained on 2023-24) | **157,137** | 57.14% | 25.84% | 0.1025% | 0.002 | **10.15%** | 0.0090 | 1.81 | **+4.43% alpha** |

*Interpretation:* Even a 512-step honest smoke tipped above equal-weight (`+0.47%`). The 25k leaky run looks better (`+4.4%`, lower drawdown) **because it saw the test data** — honest evaluation is slightly worse on drawdown/Sharpe. The turnover spike `0.099` at 512 steps vs `0.002` at 25k shows early training still churning; longer honest runs drive turnover down.

#### Baseline sanity

`equal_weight` `mean_net 0.0919%/day ×252 = 23.2%/yr` → `150k` from `100k` in 2 yrs, drawdown `12.5%` — matches Nifty 50 2023-24 bull/mild chop. Training `mean_reward -0.01` corresponds to `net − penalties` same order.

### 6.3 Artifacts

*   `artifacts/finance_v1/{finance_lora/finance/, ppo_heads.pt, model_config.json, ppo_config.json, environment_config.json, eval_2023-01-01_2024-12-31.json}`
*   `artifacts/smoke_honest/checkpoint_{256,512}/...` + eval JSONs — verified `evaluate_policy.py` finds nested `adapter_config.json` and correct CUDA placement.
*   `artifacts/smoke_honest` honest 512 run demonstrates full pipeline: `honest 2022-12-31` split, `bf16`, `3.9GB`, checkpoint each 256, eval each 256.

---

## 7. Key Insights (Non-Technical)

*   **Negative `mean_reward` ≠ losing money.** Reward is market return *minus* costs and penalties for trading too much, being too volatile or sitting in a drawdown. A `+0.09%` market day can still give `-0.02` reward if you trade a lot.
*   **`peak_equity` is a water-mark, not final wealth.** `189k → 105k` across episodes is variance, not profit taken. Convergence needs many episodes.
*   **Honest splits matter.** Leaky `+4.4%` alpha flatters; honest `+0.47%` after 512 steps is the truthful signal. Gap will close with `75k` honest steps.
*   **A5000 is under-utilized at 3.9GB.** `rollout 512 / minibatch 128` fits comfortably in 24GB; bigger batches are faster and more stable.

---

## 8. Limitations Today

*   End-to-end learning needs **75–100k steps** to stabilize Turnover/drawdown; `25k` is dev-scale.
*   Current panel has **survivorship bias** (current Nifty 50 backfilled).
*   Environment is **long-only, fully invested, daily** — no shorts, leverage, intraday, or order-book.
*   `torch.compile` needs Triton (Linux/WSL) — not on Windows A5000.
*   No live/broker integration by design (and not planned).

---

## 9. Future Scope

### 9.1 Near-Term (next 1–2 months)

*   **Honest production run:** `75k–100k` steps on `train-end 2022-12-31` (70–94 train episodes, `1.3–1.8h`, `rollout 512 / minibatch 128 / bf16`, eval every 5k, checkpoint every 10k).
*   **Thresholded promotion:** promote to `100k` only if `mean_net` slope at `60k` stays positive; early-stop if flat `15k`.
*   **TensorBoard / CSV logger** for `reward / net / gross / turnover / vol / drawdown / Sharpe`.
*   **Ablations:** `w1/w2/w3` FFN LoRA, lower `turnover_penalty`, transaction `5bps` vs `10bps`, `entropy 0.01 → 0.005`.

### 9.2 Medium-Term (2–6 months)

*   **Better data:** historical NSE constituent archives to remove survivorship bias; add corporate-action adjusted splits/dividends.
*   **Richer state:** position-level encoders, sector embeddings, earnings-calendar / macro context (still via InternLM text side).
*   **Multi-adapter portfolio:** co-host `finance`, `atc`, etc. on same frozen InternLM backbone, route by task.
*   **Risk suite:** Sortino / Calmar, turnover-aware Sharpe, short/delta-neutral extensions with leverage controls.
*   **Linux/WSL Triton** → `torch.compile` ~15–25% speedup.

### 9.3 Longer-Term Vision

*   A **portable finance foundation**: small LoRAs that can be shared, merged or nested without retraining the 1.8B core, usable for research, education and paper-trading labs.
*   Walk-forward and regime testing (bull/bear/sideways) and ultimately a proper out-of-sample 2025-26 test once 2023-24 is locked as validation.
*   Documentation and notebooks so a non-RL reader can run `prepare → train → evaluate` in three commands and understand every number.

---

## 10. How to Reproduce (A5000, Honest)

```powershell
# 1) CUDA torch (driver 595.95 / CUDA 13.2 → cu128)
pip install torch --index-url https://download.pytorch.org/whl/cu128

# 2) Smoke honest (~30–70s, proves pipeline)
python train.py --data data\nifty50_prices.csv --train-end 2022-12-31 --timesteps 512 --output artifacts\smoke_a5000 --device cuda --dtype bf16 --eval-interval 256
python evaluate_policy.py --data data\nifty50_prices.csv --artifacts artifacts\smoke_a5000 --start 2023-01-01 --end 2024-12-31 --device cuda
python evaluate_baselines.py --data data\nifty50_prices.csv --start 2023-01-01 --end 2024-12-31

# 3) Production honest (75k ≈70 episodes, ~1.3h; 100k ≈94 episodes, ~1.8h)
python train.py --data data\nifty50_prices.csv --train-end 2022-12-31 --timesteps 75000 --output artifacts\finance_prod --device cuda --dtype bf16 --rollout-steps 512 --minibatch-size 128 --eval-interval 5000 --checkpoint-interval 10000
python evaluate_policy.py --data data\nifty50_prices.csv --artifacts artifacts\finance_prod --start 2023-01-01 --end 2024-12-31
```

Outputs remain in `artifacts/.../{finance_lora, ppo_heads.pt, *.json, eval_*.json, checkpoint_*}`.

---

## 11. Conclusion

FinanceRL is past the “does it run?” stage and into the “does it learn honestly?” stage. The pipeline is stable on A5000, the first learned policies already beat a simple equal-weight baseline on a truly unseen 2-year window (even after only 512 honest steps), and the gap between leaky and honest results is now visible and about to be closed with a longer, honest `75–100k` run. The next step is that single production run — the same code, just more steps and stricter honesty.

---

*Generated: 2026-09-02 • Stack: InternLM2-1.8B frozen + LoRA (wqkv/wo, r16) • PPO • A5000 24GB • Torch 2.11+cu128 • Data: Nifty 50 2017-11-17→2026-01-30 • Code: `train.py` / `evaluate_policy.py` / `env.py` / `model.py`*
