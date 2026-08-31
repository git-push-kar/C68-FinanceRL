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

Use a local historical daily-data CSV with `Date,Ticker,Close`; `Volume` is
optional. One row is one asset on one trading day.

```csv
Date,Ticker,Close,Volume
2018-01-02,SPY,268.77,86655700
2018-01-02,QQQ,158.49,32540000
```

## Chronological workflow

Do not tune settings on the held-out period. Train through a fixed date:

For the Nifty 50, first download and normalize the daily market panel. The
script reads the current official constituent list from NSE, appends Yahoo
Finance's `.NS` symbol suffix, and produces the format expected by training:

```powershell
python prepare_nifty50_data.py --start 2020-01-01 --end 2025-01-01 --output data\nifty50_prices.csv
```

Then reserve later dates as an untouched test period:

```powershell
python train.py --data data\nifty50_prices.csv --train-end 2022-12-31 --timesteps 25000 --output artifacts\finance_v1
```

Evaluate the untouched period against an equal-weight baseline:

```powershell
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
