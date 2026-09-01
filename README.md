# Standalone Finance LoRA RL adapter

This folder is portable: copy the entire `finance_adapter` directory to a
separate project. It has no dependency on this repository's ATC code and never
connects to a broker or places live orders.

```text
finance_adapter/
├── config.py, config.yaml       # LoRA and PPO settings
├── env.py                       # historical finance environment
├── model.py                     # InternLM + named finance LoRA
├── ppo.py                       # PPO and GAE implementation
├── train.py                     # standalone training entry point
├── prepare_nifty50_data.py       # yfinance Nifty 50 data preparation
├── evaluate_baselines.py        # untouched-period baseline
├── data/                        # put local source CSVs here
├── artifacts/                   # generated LoRA and PPO outputs
└── tests/                       # smoke-test instructions
```

## Setup

```powershell
cd finance_adapter
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Dataset is **bundled** in `data/nifty50_prices.csv` (14MB cleaned panel: 287k rows, 49 Nifty tickers, 1999-2026, effective 2017-11-17..2026-01-30 after `ffill` : `env.py:45`) so `git clone` works offline. Verify:

```powershell
dir data                          # -> nifty50_prices.csv + nifty50_summary_statistics.csv
python evaluate_baselines.py --data data\nifty50_prices.csv --start 2023-01-01 --end 2024-12-31
```

To rebuild data from scratch or use `Date,Ticker,Close` + optional `Volume` (one row per asset per day):

```csv
Date,Ticker,Close,Volume
2018-01-02,SPY,268.77,86655700
2018-01-02,QQQ,158.49,32540000
```

## Chronological workflow

Do not tune settings on the held-out period. Train through a fixed date.

A5000 (Ampere, 24GB) – BF16/TF32 optimized (`train.py:34`, `model.py:21`):

```powershell
# CUDA torch first: pip install torch --index-url https://download.pytorch.org/whl/cu121
python train.py --data data\nifty50_prices.csv --train-end 2022-12-31 --timesteps 25000 --output artifacts\finance_v1 --device cuda --dtype bf16 --minibatch-size 128 --rollout-steps 512
# smoke
python train.py --data data\nifty50_prices.csv --train-end 2022-12-31 --timesteps 256 --output artifacts\smoke --device cuda --dtype bf16
```

CPU or rebuild-data fallback:

```powershell
# Re-download Nifty 50 panel (current NSE list + yfinance .NS) if you delete bundled data
python prepare_nifty50_data.py --start 2020-01-01 --end 2025-01-01 --output data\nifty50_prices.csv

python train.py --data data\nifty50_prices.csv --train-end 2022-12-31 --timesteps 25000 --output artifacts\finance_v1
python evaluate_baselines.py --data data\nifty50_prices.csv --start 2023-01-01 --end 2024-12-31
```

Artifacts are saved as `finance_lora/`, `ppo_heads.pt`, and JSON configs. Keep
the LoRA unmerged; it can later be loaded as the named `finance` adapter on the
same shared InternLM backbone as other adapters.

The environment is long-only and fully invested. Its reward is net portfolio
return minus transaction cost, turnover, volatility, and drawdown penalties.
This is research/paper-trading software, not investment advice or a live
execution system.

Nifty 50 membership changes over time. This downloader intentionally uses the
current official list for a convenient first experiment; it is not suitable for
claims about historical strategy performance because it introduces survivorship
bias. Use historical NSE constituent archives for rigorous research.
