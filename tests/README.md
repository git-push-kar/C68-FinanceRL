# Validation checklist

After installing requirements, validate a copied package in this order:

```powershell
python evaluate_baselines.py --data data\prices.csv --start 2023-01-01 --end 2024-12-31
python train.py --data data\prices.csv --train-end 2022-12-31 --timesteps 256 --output artifacts\smoke
```

The smoke run should create `artifacts/smoke/finance_lora`, `ppo_heads.pt`, and
three JSON configuration files. Do not interpret smoke-run returns as a trading
result; it only checks that data, model, PPO, and artifact persistence connect.
