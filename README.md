# FinanceRL — AI Paper-Trading on Nifty 50 (InternLM + PPO)

> **Teach a frozen 1.8B language model to manage 49 stocks — no live market, no broker, purely simulated history.**

This folder is **portable**: copy the entire `C68-FinanceRL` directory elsewhere and it runs. It has no dependency on outside ATC code.

```text
C68-FinanceRL/
├── config.py, config.yaml        # LoRA (wqkv/wo r16) + PPO (γ0.99, clip0.2, 4 epochs)
├── env.py                        # Historical, cost-aware PortfolioEnv (window 30, 10bps, 30% cap)
├── model.py                      # InternLM2-1.8B frozen + FinanceEncoders + LoRA finance + Actor/Critic
├── ppo.py                        # GAE + PPO clipped update (bf16/fp16 branches)
├── train.py                      # A5000 (bf16/TF32) training, checkpoints, periodic holdout eval
├── evaluate_policy.py            # NEW — loads any artifacts/finance_lora + ppo_heads.pt, blind holdout vs baseline
├── evaluate_baselines.py         # Equal-weight (1/49) baseline
├── prepare_nifty50_data.py       # yfinance Nifty 50 .NS downloader
├── data/                         # Bundled nifty50_prices.csv (287k rows → 2027 days) + summary
├── artifacts/                    # finance_lora/, ppo_heads.pt, *.json, eval_*.json, checkpoint_*/
├── REPORT.md / RUN_REPORT.md     # Full reports (₹1 lakh analogy, tables, A5000 proof)
└── tests/                        # Smoke instructions
```

---

## What is happening in this project

1.  **Goal — a portable finance adapter, not a full model.** We keep the shared **InternLM2-1.8B** backbone frozen and only train a tiny **LoRA `finance` (rank 16 on `wqkv/wo`)** plus three small translators. The resulting `finance_lora/` can be dropped onto the same backbone as other adapters (ATC etc.) — merge-free, `adapter_name="finance"`.

2.  **Task — daily portfolio management, long-only.** Each simulated day the agent decides weights for 49 Nifty 50 stocks (fully invested, capped at 30% per stock). It pays **10 bps transaction cost** on turnover and is judged on **net return minus penalties for turnover, volatility and drawdown** (`env.py:63`).

3.  **Honesty guarantee — chronological split.** We train only up to a cutoff date (e.g., `2022-12-31`) and test on a *completely unseen* period (`2023-01-01 → 2024-12-31`, 459 trading days) that is never used for tuning. A prior 25k run intentionally saw the test period (`finance_v1`, **leaky**) — our honest smokes and the `-6.13%` run prove the pipeline now catches leakage.

4.  **Status Sep 2026 — fully working on RTX A5000 (24GB) in bf16 (≈3.9GB VRAM).** Three live holdouts on the same blind 2-yr window:
    *   `equal_weight` (do nothing) → **₹1,50,471** from ₹1L
    *   honest smoke 512 steps → **₹1,51,187 (+0.47%)**
    *   leaky 25k (saw future) → **₹1,57,138 (+4.43%)** — flattered
    *   honest longer (you just shared) → **₹1,41,251 (-6.13%)**, `22.4%` drawdown — too wobbly, high vol `0.014` vs `0.0076` baseline. All numbers from `artifacts/*/eval_2023-01-01_2024-12-31.json`.

    The platform is now past “does it run?” and into “does it learn honestly?” — with `75k–100k` honest production runs wired (`1.3–1.8h`).

> **Not investment advice.** Research / paper-trading simulated on historical prices.

---

## Architecture & Approach

### System overview

```
data/nifty50_prices.csv ──► PortfolioEnv (window 30, 10bps, 30% cap) ──► {market, portfolio, risk}
                                                       │
                                                       ▼
                                          FinanceEncoders (3× MLP, 2048-d)
                                          • market: Linear(5→2048)→GELU→Linear
                                          • portfolio: LayerNorm(50)→Linear
                                          • risk: LayerNorm(3)→Linear
                                                       │  stack → 3 tokens × 2048
                                                       ▼  as inputs_embeds
                                   InternLM2-1.8B frozen (24 layers, hidden 2048, bf16)
                                         + LoRA finance (wqkv/wo, r=16, α=32, dropout 0.05)
                                                       │  hidden_states[-1][:,-1]  (use_cache=False)
                                                       ▼
                                   Actor Linear(2048→49) ─► weights (softmax + cap)
                                   Critic Linear(2048→1) ─► value
                                   log_std (49) ─► Gaussian policy
                                                       └──── PPO + GAE (4 epochs) ────┘
```

### Approach — why this way

*   **Frozen LLM as sequence prior:** Rather than training a transformer from scratch on scarce price data, we reuse InternLM’s sequence modeling. The 1.8B model is never updated — only the 3 encoders + LoRA + heads learn.
*   **Encoders translate numbers to LLM tokens:** Market (5 features × 30 days × 49 stocks), portfolio (49+1), and risk (3) each become a 2048-d token. Three tokens are fed as `inputs_embeds` — the LLM attends over market/portfolio/risk like a 3-token sentence.
*   **PPO for the hard part:** No supervised label exists. PPO explores with a Gaussian (`log_std ~ -0.5`), collects rollouts (256/512 steps), computes advantages via GAE (`γ0.99, λ0.95`), and updates with a clipped surrogate (`clip 0.2, value 0.5, entropy 0.01, 4 epochs`). Advantages are normalized; `train.py` handles `bf16` autocast and `fp16` GradScaler branches.
*   **Reward shaping for real frictions:** `net = gross - turnover×10bps`, `reward = net - 0.001×turnover - 0.10×20d_vol - 0.20×drawdown`. This makes a `+0.09%/day` market day give `-0.02` reward if you churn — explaining why `mean_reward` in logs is negative even when the portfolio grows.
*   **InternLM specifics fixed:** Auto-remap `q_proj/v_proj → wqkv/wo`, `use_cache=False` + `output_hidden_states=True` (`CausalLMOutput` on `transformers 5.16`), `bf16` cast for obs vs encoders, `bf16→float→numpy` for env stepping, `float()` for GAE stability.
*   **Efficiency on A5000 (Ampere sm_86):** `bf16` native (2× vs `fp32`), `TF32` matmul + cuDNN `TF32`, `high` matmul precision, autocast per step. `3.9GB` alloc / `6.05GB` reserved for 49 assets; `triton` missing on Windows so `torch.compile` falls back — needs Linux/WSL.

### Data & Environment details

*   **Bundled panel** `data/nifty50_prices.csv`: 14 MB, 287,263 long-format rows, 49 tickers (current NSE list `.NS`), `1999-01-01→2026-01-30` raw → **effective `2017-11-17→2026-01-30` after `pivot().ffill().dropna()` (`env.py:45`) → 2,027 trading days** (`train.py` logs `dates=2027`; honest `1266` after `train-end 2022-12-31`). `nifty50_summary_statistics.csv` provides per-ticker total return (`EICHERMOT 998k%` etc.), vol, splits/dividends.
*   **Window 30:** Each obs is `market (30×49×5)`, `portfolio (50)`, `risk (3)`.
*   **Cap loop (`env.py:70-77`):** scores → soft-max → cap at `30%`, redistribute excess — guarantees diversification.
*   **Episode:** `reset()` sets `index=window`, `step()` advances one day, terminates at `len(prices)-2`; one episode `≈ 2027-30-2 = 1995` steps full panel, `~1068` honest.

### Files — what does what

| File | Role |
|------|------|
| `config.py` | `ModelConfig` (`internlm2_5-1_8b-chat`, `wqkv/wo`, `r16`, `bf16`) + `PPOConfig` (`rollout 256→512`, `minibatch 64→128` prod recipe `75–100k` comments) |
| `env.py` | PortfolioEnv, FinanceRewardConfig |
| `model.py` | FinanceEncoders + InternLMFinancePolicy + `save_artifacts()` |
| `ppo.py` | `compute_gae`, `ppo_update` |
| `train.py` | A5000 loop, detailed logging `mean_reward/mean_net/max_dd/turnover/peak_equity`, `checkpoint_<steps>/`, periodic `[Holdout ...]` when `--train-end` set, Triton fallback |
| `evaluate_policy.py` | **New** — loads any `artifacts/...` (handles `finance_lora/finance/adapter_config.json` nested) onto CUDA, runs deterministic `mean` policy on any `start→end`, reports `final/peak/return/net/gross/turnover/cost/dd/vol/Sharpe/alpha` vs equal-weight, saves `eval_*.json` |
| `evaluate_baselines.py` | Equal-weight `1/49` baseline |
| `data/` / `artifacts/` / `prepare_nifty50_data.py` | Bundled data, generated LoRAs/heads/JSONs/checkpoints, yfinance builder |

---

## Setup

```powershell
cd C68-FinanceRL
py -3.11 -m venv venv; .\venv\Scripts\Activate.ps1
pip install -r requirements.txt      # torch≥2.4, transformers≥4.45, peft≥0.13, accelerate, yfinance

# For A5000 (driver 595.95 / CUDA 13.2 tested: cu128 → 3.9GB, sm_86 25.8GB):
pip install torch --index-url https://download.pytorch.org/whl/cu128  # verified 2.11+cu128
```

Verify data & baseline:

```powershell
dir data                         # nifty50_prices.csv + nifty50_summary_statistics.csv
python evaluate_baselines.py --data data\nifty50_prices.csv --start 2023-01-01 --end 2024-12-31
# {baseline: equal_weight, final_equity: 150471, mean_net_return: 0.00091, max_drawdown: 0.1246}
```

Rebuild data from scratch (optional, needs `Volume` column):

```csv
Date,Ticker,Close,Volume
2018-01-02,SPY,268.77,86655700
2018-01-02,QQQ,158.49,32540000
```
```powershell
python prepare_nifty50_data.py --start 2020-01-01 --end 2025-01-01 --output data\nifty50_prices.csv
```

---

## Chronological Workflow — Do not tune on holdout

### A5000 honest workflow (BF16/TF32, `train.py:45`, `model.py:22`)

```powershell
# CUDA torch first (cu128):
pip install torch --index-url https://download.pytorch.org/whl/cu128

# Smoke honest (~30–70s, proves pipeline + holdout eval, 512 steps):
python train.py --data data\nifty50_prices.csv --train-end 2022-12-31 --timesteps 512 --output artifacts\smoke_a5000 --device cuda --dtype bf16 --eval-interval 256
python evaluate_policy.py --data data\nifty50_prices.csv --artifacts artifacts\smoke_a5000 --start 2023-01-01 --end 2024-12-31 --device cuda
python evaluate_baselines.py --data data\nifty50_prices.csv --start 2023-01-01 --end 2024-12-31
# Smoke gave honest +0.47% (151,187 vs 150,471)

# Production honest (75k ≈70 episodes ~1.3h; 100k ≈94 episodes ~1.8h):
python train.py --data data\nifty50_prices.csv --train-end 2022-12-31 --timesteps 75000 --output artifacts\finance_prod --device cuda --dtype bf16 --rollout-steps 512 --minibatch-size 128 --eval-interval 5000 --checkpoint-interval 10000
# or 100k:
python train.py --data data\nifty50_prices.csv --train-end 2022-12-31 --timesteps 100000 --output artifacts\finance_prod_100k --device cuda --dtype bf16 --rollout-steps 512 --minibatch-size 128 --eval-interval 5000
python evaluate_policy.py --data data\nifty50_prices.csv --artifacts artifacts\finance_prod --start 2023-01-01 --end 2024-12-31 --device cuda

# Leaky vs honest comparison (same holdout, shows bias):
python evaluate_policy.py --data data\nifty50_prices.csv --artifacts artifacts\finance_v1 --start 2023-01-01 --end 2024-12-31 --device cuda   # leaky 157,138
python evaluate_policy.py --data data\nifty50_prices.csv --artifacts artifacts\smoke_honest\checkpoint_512 --start 2023-01-01 --end 2024-12-31 --device cuda  # honest smoke 151,187
```

### CPU / fallback

```powershell
python train.py --data data\nifty50_prices.csv --train-end 2022-12-31 --timesteps 25000 --output artifacts\finance_v1
python evaluate_baselines.py --data data\nifty50_prices.csv --start 2023-01-01 --end 2024-12-31
```

Training logs `mean_reward / mean_net / max_dd / mean_turnover / peak_equity` per rollout + VRAM, plus periodic `[Holdout ...] learned vs equal_weight` when `--train-end` is set. Checkpoints land in `artifacts/.../checkpoint_<steps>/`.

Artifacts are `finance_lora/` (holds `finance/adapter_config.json` nested) + `ppo_heads.pt` + `model/ppo/environment_config.json` + `eval_*.json`. Keep LoRA unmerged for reuse on the shared InternLM backbone. Full report with ₹1 lakh tables: `REPORT.md` / `RUN_REPORT.md`.

### Environment note

*Reward is not market return:* `net = gross - turnover×10bps`, `reward = net - 0.001×turnover - 0.10×vol - 0.20×drawdown`. So negative `mean_reward` is normal even when equity grows. Nifty 50 list is **current constituents backfilled** → **survivorship bias** — use NSE constituent archives for rigorous history.

### Hardware proof

`RTX A5000, WDDM, sm_86, 24,564 MiB, torch 2.11+cu128, cuda available, bf16 (~3.9GB alloc)`, InternLM2 1.8B frozen, `wqkv/wo` LoRA. See `REPORT.md §5.2`, `RUN_REPORT.md` ₹1 lakh tables.

---

*Research / paper-trading, not investment advice. Generated Sep 2 2026 — Stack InternLM2-1.8B + LoRA wqkv/wo r16, PPO, A5000 24GB, Torch 2.11+cu128, Data Nifty 50 2017-11-17→2026-01-30.*
