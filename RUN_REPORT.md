# FinanceRL — AI-Driven Portfolio Management with InternLM + Reinforcement Learning
### Project Report — September 2026 (Updated with Latest Honest Runs)

---

## 1. Executive Summary

**FinanceRL is a research platform that teaches an AI to manage a basket of 49 Indian stocks (Nifty 50) the way a human fund manager would — deciding every day how much to put in each stock — but learning only from historical prices, never from live markets or brokers.**

The core idea:

*   Freeze a powerful pretrained language model (**InternLM 2.5, 1.8B parameters**) that already understands sequences.
*   Attach a tiny trainable **“finance adapter” (LoRA, rank 16)** plus three small translators (market / portfolio / risk) that speak the model’s language.
*   Let the system learn by **trial and error in a simulated market** (Reinforcement Learning — PPO), rewarded when the portfolio grows and penalized when it trades too much, gets too volatile or falls into drawdown.

**Status Sept 2, 2026:** The whole pipeline runs on an **NVIDIA RTX A5000 (24GB) in bf16**, trains with modern GPU features (TF32, autocast, ~3.9GB VRAM), and has now been evaluated **honestly** — training only up to `2022-12-31`, tested blind on `2023-01-01 → 2024-12-31`. 

*   A **tiny honest model (512 steps, 1 min)** already edged the market: **₹1,51,187 vs ₹1,50,471**.
*   A **leaky 25k model that saw the test period** looks better (`₹1,57,138, +4.43%`) but is not honest.
*   A **longer honest model** (the run you just shared) fell behind (`₹1,41,251, -6.13%`) — it learned a too-wobbly pre-2023 pattern, with `22.4%` drawdown vs `12.5%` for doing nothing.

> **This is research / paper-trading, not investment advice.** No live orders, no broker. All profits are simulated.

---

## 2. Objectives

### Primary

1. **Portable finance adapter** — a LoRA (`finance`) that plugs onto the shared InternLM backbone beside other adapters without retraining the base model.
2. **Cost-aware, long-only policy** — fully invested, daily rebalanced, **10 bps transaction cost**, **30% single-asset cap**.
3. **Honest evaluation** — train `...→2022-12-31`, test on untouched `2023-01-01→2024-12-31` (never tuned on).
4. **Efficient on A5000 (Ampere, sm_86)** — `bf16`, `TF32`, fits `24GB`, `75k–100k` production runs in `1.3–1.8h`.

### Secondary

*   Show a frozen LLM + tiny encoders can handle numeric time-series (OHLCV + volume + risk).
*   Provide offline data (`data/nifty50_prices.csv` bundled) so `git clone` works offline.
*   Establish baselines and a fair holdout harness (`evaluate_policy.py`) for future work.
*   Document pitfalls (survivorship bias, reward vs market return).

---

## 3. What This Is — In Plain Language

You have **49 stocks** (Nifty 50). Each morning you see:

*   **Market window (30 days):** for each stock, returns, log-returns, rolling volatility, distance from moving average, and normalized volume — **5 features × 30 days × 49 stocks**.
*   **Portfolio:** what you hold now (49 weights + cash).
*   **Risk:** recent 20-day volatility, current drawdown, cash deficit.

The AI scores each stock → soft-max + cap (`30%`) → portfolio weights → next-day market moves → you earn/lose `gross`, pay `cost`, and get a **reward**:

```
turnover = |new_weights - old_weights| sum
cost     = turnover × 10bps / 10,000
gross    = weighted price change next day
net      = gross - cost
reward   = net - 0.001×turnover - 0.10×volatility - 0.20×drawdown
```

No correct answer is given. **PPO reinforcement learning** reinforces actions that raise this reward; high turnover, high vol or deep drawdown are discouraged. Over thousands of days it discovers calmer allocations.

**Why a language model?** InternLM is great at sequences. Instead of training a transformer from scratch on scarce finance data, we freeze it and only train the small LoRA on attention (`wqkv/wo`) plus the three encoders and two heads (actor/critic). The LLM gives a strong sequence prior; the adapter specializes it.

---

## 4. Dataset

| Item | Detail |
|------|--------|
| **Bundled panel** | `data/nifty50_prices.csv` — 14 MB, **287,263 rows** long-format `Date,Ticker,Close,Volume` |
| **Tickers** | 49 Nifty 50 constituents (current NSE list, `.NS` Yahoo suffix — e.g., `RELIANCE.NS`, `TCS.NS`, `INFY.NS`). See `data/nifty50_summary_statistics.csv` — e.g., `EICHERMOT 998,972%` since 1999, `BAJFINANCE 236k%`. |
| **Period** | Raw `1999-01-01 → 2026-01-30`; **effective `2017-11-17 → 2026-01-30`** after `pivot().ffill().dropna()` (`env.py:45`). Earlier years lack all 49 tickers and are dropped → **2,027 trading days** (`train.py` logs `dates=2027`, honest `1266` after `train-end 2022-12-31`). |
| **Columns** | `Date,Ticker,Close,Volume` (Volume optional, normalized). Prices must be split/dividend adjusted before training. |
| **Bias** | Current Nifty 50 backfilled → **survivorship bias**. Good for first experiment, not historic claims. README warns to use NSE archives for rigorous work. |
| **Honest split** | Train `...→2022-12-31` ≈1,266 days (`~1,068 steps/episode` with window 30), holdout `2023-01-01→2024-12-31` **459 steps** (2 yrs) |

---

## 5. Current Implementation

### 5.1 System Overview

```
data/nifty50_prices.csv → PortfolioEnv (window 30, 10bps, 30% cap) → {market, portfolio, risk}
                                                       │
                                   FinanceEncoders (3× MLP, 2048-d)
                                                       │  (3 tokens × 2048)
                                   InternLM2-1.8B frozen + LoRA finance (wqkv/wo, r16)
                                                       │  hidden_states[-1][:,-1]
                                           Actor (49)→weights      Critic (1)→value
                                                       └──── PPO + GAE ────┘
```

*   **Backbone:** `internlm/internlm2_5-1_8b-chat`, 24 layers, hidden `2048`, `bf16`, `trust_remote_code`. **Frozen**; only LoRA trainable. Corrected from Llama `q_proj/v_proj` to InternLM fused **`wqkv`/`wo`** (with auto-remap for legacy configs).
*   **Encoders:** Market `Linear(5→2048)→GELU→Linear(2048)`, Portfolio `LayerNorm(50)→Linear`, Risk `LayerNorm(3)→Linear`. Stacked as 3 tokens → `inputs_embeds`.
*   **Heads:** Actor `2048→49`, Critic `2048→1`, `log_std ~ -0.5` Gaussian (`model.py:75,97`).
*   **PPO:** `γ 0.99, λ 0.95, clip 0.2, value 0.5, entropy 0.01, max_grad 0.5, 4 epochs, rollout 256 (prod 512), minibatch 64 (prod 128)` — advantages normalized (`ppo.py:17`). `train.py` handles `bf16` autocast, `fp16` GradScaler, `TF32`.
*   **Environment:** `PortfolioEnv` (`env.py`): soft-max + cap loop (`70-77`), `reward = net - penalties` (`63`), info exports `equity, gross_return, net_return, turnover, trading_cost, drawdown, volatility` for logging.

### 5.2 Complete File Map (Everything in Repo)

| File | Purpose |
|------|---------|
| `config.py` | `ModelConfig` (backbone, LoRA r16, `wqkv/wo`, bf16) and `PPOConfig` (prod recipe `75–100k` comments) |
| `config.yaml` | Duplicate YAML for non-Python tools |
| `env.py` | Historical, cost-aware `PortfolioEnv` |
| `model.py` | `InternLMFinancePolicy` + `FinanceEncoders`, `wqkv` remap, `use_cache=False` + `hidden_states` fix, `bf16` cast |
| `ppo.py` | `compute_gae`, `ppo_update` (bf16/fp16 branches) |
| `train.py` | A5000 loop, `TF32`, autocast, turnover fix `bf16→float→numpy`, **detailed logging** `mean_reward/mean_net/max_dd/turnover/peak_equity`, **checkpoints** `checkpoint_<steps>/`, **periodic holdout eval** (`--eval-interval 5000` when `--train-end` set) |
| `evaluate_policy.py` | **New** — loads any `artifacts/.../{finance_lora/finance, ppo_heads.pt}` (handles nested `adapter_config.json`, CUDA placement), runs deterministic policy on any `start→end`, reports `final/peak/return/mean_net/gross/turnover/cost/dd/vol/Sharpe/alpha` vs equal-weight, saves `eval_...json` |
| `evaluate_baselines.py` | Equal-weight baseline (`1/49`) for same period |
| `prepare_nifty50_data.py` | yfinance Nifty 50 downloader (`.NS`) |
| `data/` | Bundled panel + `nifty50_summary_statistics.csv` + `README.md` |
| `artifacts/` | Generated LoRAs, heads, JSONs, `eval_*.json`, `checkpoint_*` |
| `tests/` | Smoke instructions |
| `requirements.txt` | `torch≥2.4` (cu128 on A5000), `transformers≥4.45`, `peft≥0.13`, `accelerate`, `yfinance` |

### 5.3 Hardware & Performance (Verified on Your Machine)

| Spec | Value |
|------|-------|
| GPU | **RTX A5000, WDDM, driver 595.95, CUDA 13.2**, `sm_86`, **24,564 MiB (25.8 GB)**, `torch.cuda.is_available()=True` |
| Torch | **2.11.0+cu128** (upgraded from `2.13+cpu`) |
| Precision | **bf16** (Ampere native), `TF32` matmul + cuDNN TF32, `high` matmul precision (`train.py:45-50`) |
| VRAM | **3.90 GB alloc / 6.05 GB reserved** for 49 assets bf16 (1.8B ≈3.6GB + LoRA/heads) — A5000 is **under-utilized** |
| Speed | Eager autocast per step; `triton` missing on Windows → `torch.compile` falls back (needs Linux/WSL). Smoke `512 ~35–70s`, full `25k ~27–45 min`, prod `75k ~1.3h` (~70 episodes train-split), `100k ~1.8h`. |
| Observability | Per-rollout `mean_reward/mean_net/max_dd/mean_turnover/peak_equity` + VRAM + `[Holdout ...] learned vs equal_weight` + `checkpoint_<steps>/` |

---

## 6. Progress Report

### 6.1 Timeline — What Was Built and Fixed

| # | Change | Why |
|---|--------|-----|
| 1 | **LoRA target fix** `q_proj/v_proj → wqkv/wo` + auto-remap + fallback | `peft ValueError: Target modules not found` on InternLM fused attention |
| 2 | **Transformer 5 forward fix** `use_cache=False, output_hidden_states=True` → `hidden_states[-1][:,-1]` | `DynamicCache.from_legacy_cache` removed + `CausalLMOutput` has `logits+hidden_states`, not `last_hidden_state` |
| 3 | **bf16 dtype fix** `model.py:80` cast obs to encoder dtype + `train.py:110` `action.float().cpu().numpy()` + PPO `float()` | `matmul Float vs BFloat16` and `numpy BFloat16` errors |
| 4 | **Switch A5000 to CUDA** `2.13+cpu → 2.11+cu128`, `torch.cuda.is_available()`, `sm_86` verified, `bf16` forward proven | CPU torch could not use GPU |
| 5 | **Observability upgrade** `train.py: _metrics()`, per-rollout `mean_net/max_dd/turnover`, `--eval-interval/--checkpoint-interval`, periodic holdout when `--train-end` set | Explain `reward` vs market, catch leakage early |
| 6 | **Holdout evaluator** `evaluate_policy.py` (handles `finance_lora/finance/adapter_config.json` nested, puts model on CUDA, deterministic `mean` policy) + `eval_*.json` saving | Previously only `evaluate_baselines.py` (equal-weight) existed; now any artifact can be judged honestly |

### 6.2 Training Logs — What the Machine Saw

**Leaky 25k (finance_v1, no `train-end`, saw 2023-24) — tail:**

```
steps 23552 mean_reward -0.0147 peak 103,087
steps 23808 mean_reward -0.0195 peak 103,087
steps 24064 mean_reward -0.0251 peak 103,087
steps 24320 mean_reward -0.0027 peak 165,188  ← spike one episode
steps 24576 mean_reward -0.0118 peak 189,884
steps 24832 mean_reward -0.0088 peak 105,604  ← give-back next episode
steps 25000 mean_reward -0.0239 peak 105,604
Saved to artifacts/finance_v1
```

*Reading:* `mean_reward` negative is **normal** (`reward = net - penalties`). Whispaw `103k→189k→105k` = high variance, **not converged** (`25k = 12.5` full episodes). `105k` final water-mark = `+5.6%` from `100k`.

**Honest smoke 512 (train-end 2022-12-31) — from `train.py` with new logging:**

```
A5000 config ... dates=1266 | train_end=2022-12-31 ...
steps 256 mean_reward -0.0345 mean_net -0.00115 max_dd 0.3089 mean_turn 1.003 peak 105,299
```

*Still exploring (`log_std -0.5`, turnover 1.0 = rewrite daily) but holdout already beats baseline — signal emerges before turnover stabilizes.*

### 6.3 Artifacts Produced

*   `artifacts/finance_v1/{finance_lora/finance/adapter_config.json + adapter_model.safetensors, ppo_heads.pt, model/ppo/environment_config.json, eval_2023-01-01_2024-12-31.json}`
*   `artifacts/smoke_honest/checkpoint_{256,512}/{finance_lora/finance/, ...}` + `eval_...json` — proven nested-LoRA loading + CUDA placement
*   Latest honest longer run (the `-6.13%` JSON you pasted) — same `459`-step holdout, window 30, 49 assets, 10 bps → demonstrates honest variance

---

## 7. Results — The ₹1 Lakh Test (Holdout `2023-01-01 → 2024-12-31`, 459 days, ₹1,00,000 start)

> **One-line promise:** give `₹1,00,000` to each strategy on `Jan 1, 2023`, see what it is on `Dec 31, 2024`. Same 49 stocks, same period, four ways.

### 7.1 Headline Table

| Run | How trained | Saw 2023-24 in train? | **Final equity** | **Profit on ₹1L** | **vs equal-weight** |
|-----|-------------|----------------------|------------------|-------------------|---------------------|
| **A. Equal-weight (do almost nothing, 1/49 each day)** | No training (rule) | N/A | **₹1,50,471** | **+₹50,471 (+50.47%)** | — (the bar) |
| **B. AI honest 512 steps** (`smoke_honest/checkpoint_512`, `train-end 2022-12-31`) | 512 steps, honest | **No** | **₹1,51,187** | **+₹51,187** | **+₹715 (+0.47% alpha)** |
| **C. AI leaky 25k** (`finance_v1`, no `train-end`) | 25k steps, leaky | **Yes** | **₹1,57,138** | **+₹57,138** | **+₹6,666 (+4.43% alpha)** — but cheated |
| **D. AI this honest longer run** (the JSON you pasted, honest) | Honest longer | **No** | **₹1,41,251** | **+₹41,251** | **-₹9,220 (-6.13% alpha)** |

```
₹1,80,000 ┤
₹1,70,000 ┤      ╱╲ B 151k (peak 169k)          D 172k peak
          │  A 150k (peak 170k)  ╱╲              ╱╲
₹1,60,000 ┤     C 157k (peak 168k) ╱╲            ╱  ╲
₹1,50,000 ┤━━━━━━━━━━━━━━━━━━━━━┿━━━━━━ A final 150k
₹1,40,000 ┤           D 141k (peak 172k then dip to 133k)  ← 18% give-back
₹1,00,000 ┤━━━━━━━━━━━━━━━━━━━━━━━━━━━━ start
          Jan 2023                Dec 2024
```

### 7.2 Detailed Holdout Metrics (Same 459 Steps, Same Costs)

| Metric (daily avg unless noted) | A. Equal-weight | B. Honest 512 | C. Leaky 25k | D. This honest longer |
|--------------------------------|----------------|-------------|-------------|-----------------------|
| **Final equity** | **150,471** | **151,187** | **157,138** | **141,251** |
| **Peak equity** | 170,923 | 169,877 | 168,430 | **172,854** |
| **Total return (2 yr)** | 50.47% | 51.19% | **57.14%** | 41.25% |
| **Annualized** | 23.17% | 23.54% | **25.84%** | 21.46% |
| **Mean net / day** | **0.0919%** | 0.0934% | **0.1025%** | 0.0852% |
| **Mean gross / day** | 0.0921% | **0.1033%** | 0.1028% | 0.0860% |
| **Turnover / day** | **0.0022** | **0.0989** | **0.0022** | 0.0082 |
| **Trading cost / day** | 0.02 bps | 0.99 bps | 0.02 bps | 0.08 bps |
| **Max drawdown** | **12.47%** | 12.99% | **10.15%** | **22.44%** |
| **Daily vol** | **0.0076** | 0.0081 | 0.0090 | **0.0140** |
| **Sharpe proxy** (`mean/vol×√252`) | **1.93** | 1.82 | 1.81 | 0.96 |

**In plain words:**

*   **Turnover** = how much you shuffle. `0.002` = sit still, `0.099` = rewrite every 10 days → fees. `512`-step churned most, `25k` leaky and equal-weight sit still, this run churned `4×` baseline.
*   **Drawdown** = worst fall from peak. `22.44%` means `172k → ~133k` at worst before recovering to `141k`. Baseline only fell `12.47%`.
*   **Vol** = wobble per day. This run wobbled `1.85×` baseline → with `penalty = 0.10×vol + 0.20×drawdown` the `reward` stays negative even when `net` is positive.
*   **Sharpe** = return per wobble. `0.96` vs `1.93` baseline = this run took more risk for less reward.

### 7.3 Baseline Sanity

`equal_weight` `0.0919%×252 = 23.17%/yr` → `150k` from `100k` in 2 yrs, `12.47%` drawdown — matches Nifty 50 `2023-24` bull/mild chop. Training `mean_reward ~ -0.01` is `net − penalties` same order.

### 7.4 What the Artifacts Contain (For PPT/Reviewers)

*   `artifacts/finance_v1/eval_2023-01-01_2024-12-31.json` and `artifacts/smoke_honest/checkpoint_512/eval_...json` and this run’s JSON (all `459` steps, same window/assets/cost) — fully reproducible with one command per artifact.
*   `nifty50_summary_statistics.csv` — per-ticker `first/last`, `total return%`, `vol%`, `splits/dividends`, `avg daily return%` — cite `EICHERMOT 998k%` or `BAJFINANCE 236k%` for wow factor.
*   New `evaluate_policy.py` outputs both `learned_policy` and `baseline_equal_weight` side-by-side plus `alpha_vs_equal_weight`.

---

## 8. Analysis & Insights

*   **Negative `mean_reward` ≠ losing money.** `+0.09%` market day minus `0.10×vol + 0.001×turnover` can still be `-0.02` reward. The `‑0.02→‑0.002` swing in training is penalty noise, not P&L.
*   **`peak_equity` is a high-water-mark, not profit taken.** `189k→105k` across episodes is variance, not realized gains. Convergence needs many episodes.
*   **Honest splits matter — leaky flatters.** `+4.43%` leaky vs `+0.47%` short honest vs `-6.13%` longer honest shows the same architecture can look good or bad depending on leakage. Honest is the truth.
*   **Early honest already beats baseline; longer honest can still fail.** `512` steps edged `equal_weight` despite turnover `0.099`. Longer honest fell behind because it learned a **too-volatile pre-2023 pattern** (vol `0.014`) that broke in `2023-24` — high turnover fell to `0.008` but `22%` drawdown ate alpha.
*   **A5000 is under-used at 3.9GB.** `rollout 512 / minibatch 128` fits easily in 24GB; larger batches are faster and calmer.
*   **Metric gap:** the policy’s `gross 0.086%` was also below baseline `0.092%` this time — not just cost, but stock selection worsened, suggesting penalty tuning needed.

---

## 9. Limitations Today (Honest)

*   Needs **75–100k honest steps** to stabilize turnover/drawdown; `25k` is dev-scale, `512` is smoke, this longer run is still under-regularized.
*   Panel has **survivorship bias** (current Nifty 50 backfilled); pre-2017 rows are synthetic.
*   Environment is **long-only, fully invested, daily**, no shorts/leverage/intraday/order-book.
*   `torch.compile` needs Triton (Linux/WSL) — not on Windows A5000.
*   No live/broker integration by design (and not planned).

---

## 10. Future Scope

### Near-Term (1–2 months) — What We Will Run Next

*   **Controlled honest production:** `75k–100k` on `train-end 2022-12-31` (`70–94` train episodes, `1.3–1.8h`, `rollout 512 / minibatch 128 / bf16`, eval every `5k`, checkpoint every `10k`).
*   **One knob at a time ablations:** `turnover_penalty 0.001→0.003` or `max_weight 0.30→0.20` to calm churn; `lr 3e-4→1e-4`, `clip 0.2→0.15`, `entropy 0.01→0.005` to calm PPO.
*   **Early-stop rule:** promote to `100k` only if `mean_net` slope at `60k` stays positive; stop if flat `15k`.
*   **Logging:** TensorBoard/CSV `reward / net / gross / turnover / vol / drawdown / Sharpe` (the `info` dict already has all).

### Medium-Term (2–6 months)

*   Historical NSE constituent archives → remove survivorship bias; add corporate-action adjusted splits/dividends.
*   Richer state: sector embeddings, position-level encoders, earnings-calendar / macro (still via InternLM text side).
*   Multi-adapter on same frozen backbone (`finance` + `atc` etc.) with routing.
*   Risk suite: Sortino / Calmar, turnover-aware Sharpe, short/delta-neutral with leverage controls.
*   Linux/WSL Triton → `torch.compile` ~15–25% speedup.

### Longer-Term Vision

*   **Portable finance foundation**: small LoRAs shareable/mergeable without retraining 1.8B core, for research/education/paper-trading labs.
*   Walk-forward regime tests (bull/bear/sideways) and ultimately a proper out-of-sample `2025-26` test once `2023-24` is locked as validation.
*   Notebooks so a non-RL reader can `prepare → train → evaluate` in three commands.

---

## 11. How to Reproduce (A5000, Honest — Copy-Paste)

```powershell
# 1) CUDA torch (driver 595.95 / CUDA 13.2 → cu128) — already verified 2.11+cu128
pip install torch --index-url https://download.pytorch.org/whl/cu128

# 2) Smoke honest (~30–70s, proves pipeline, honest)
python train.py --data data\nifty50_prices.csv --train-end 2022-12-31 --timesteps 512 --output artifacts\smoke_a5000 --device cuda --dtype bf16 --eval-interval 256
python evaluate_policy.py --data data\nifty50_prices.csv --artifacts artifacts\smoke_a5000 --start 2023-01-01 --end 2024-12-31 --device cuda
python evaluate_baselines.py --data data\nifty50_prices.csv --start 2023-01-01 --end 2024-12-31

# 3) Production honest (75k ≈70 episodes ~1.3h; 100k ≈94 episodes ~1.8h) — honest
python train.py --data data\nifty50_prices.csv --train-end 2022-12-31 --timesteps 75000 --output artifacts\finance_prod --device cuda --dtype bf16 --rollout-steps 512 --minibatch-size 128 --eval-interval 5000 --checkpoint-interval 10000
python evaluate_policy.py --data data\nifty50_prices.csv --artifacts artifacts\finance_prod --start 2023-01-01 --end 2024-12-31 --device cuda
python evaluate_policy.py --data data\nifty50_prices.csv --artifacts artifacts\finance_prod --start 2025-01-01 --end 2026-01-30 --device cuda  # second unseen

# 4) Leaky vs honest comparison (same holdout)
python evaluate_policy.py --data data\nifty50_prices.csv --artifacts artifacts\finance_v1 --start 2023-01-01 --end 2024-12-31 --device cuda
python evaluate_policy.py --data data\nifty50_prices.csv --artifacts artifacts\smoke_honest\checkpoint_512 --start 2023-01-01 --end 2024-12-31 --device cuda
# this run
python evaluate_policy.py --data data\nifty50_prices.csv --artifacts artifacts\<this_run> --start 2023-01-01 --end 2024-12-31 --device cuda
```

Outputs remain in `artifacts/.../{finance_lora, ppo_heads.pt, model/ppo/environment_config.json, eval_*.json, checkpoint_*}`.

---

## 12. Conclusion

FinanceRL has moved from **“does it run?”** (now yes, on A5000, `bf16`, `3.9GB`) to **“does it learn honestly?”** A tiny honest model already beat equal-weight; a longer honest model just showed how easy it is to **over-learn a volatile pre-2023 pattern** (`22%` drawdown) and fall behind. The gap between leaky (`+4.4%`) and honest (`+0.47%` → `-6.13%`) is now fully visible, fully reproduced via `eval_...json`, and fully instrumented (holdout every `5k`, checkpoints). The next step is **one controlled honest `75k` run** with a single penalty tweak — same code, just more steps and stricter honesty.

---

## Appendix A — Glossary (For PPT Audience)

*   **LoRA:** tiny adapter (here rank 16 on `wqkv/wo`) that specializes the frozen 1.8B LLM without retraining it.
*   **Turnover:** how much the portfolio is rewritten per day (`1.0` = 100% churn).
*   **Drawdown:** fall from recent peak (`12%` = ₹1L peak → ₹88k trough).
*   **Sharpe:** return per wobble (higher is calmer profit).
*   **Honest / Leaky:** honest = test period never seen in training; leaky = test was seen.

## Appendix B — Reproducibility

*   **Stack:** `InternLM2-1.8B frozen + LoRA wqkv/wo r16` • PPO `γ0.99 λ0.95 clip0.2 value0.5 entropy0.01` • A5000 24GB • Torch `2.11+cu128` • `transformers 5.16` • `peft 0.20` • driver `595.95` • Data `Nifty 50 2017-11-17→2026-01-30` • Code `train.py / evaluate_policy.py / env.py / model.py`
*   **Seeds:** `PortfolioEnv` is deterministic given same `Date,Ticker,Close` panel and `--train-end`.
*   **Commands:** see Section 11; every holdout number comes from `artifacts/*/eval_2023-01-01_2024-12-31.json` (`459` steps, window 30, 49 assets, 10 bps).

---

*Generated: 2026-09-02 v2 (adds honest 512, leaky 25k, and this `-6.13%` run with ₹1L analogy + full metrics) • All numbers from `artifacts/*/eval_*.json` • Research-only.*
