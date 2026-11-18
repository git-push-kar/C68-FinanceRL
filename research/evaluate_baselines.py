"""Standalone equal-weight baseline for the untouched finance test period."""
import argparse
import json
import numpy as np
from env import PortfolioEnv
from metrics import metrics_from_infos

if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--data", required=True); parser.add_argument("--start", required=True); parser.add_argument("--end", required=True); parser.add_argument("--window", type=int, default=30); args = parser.parse_args()
    env = PortfolioEnv(args.data, window=args.window, start_date=args.start, end_date=args.end); obs, _ = env.reset(); metrics = []
    while True:
        obs, _, terminated, truncated, info = env.step(np.zeros(env.n_assets, dtype=np.float32)); metrics.append(info)
        if terminated or truncated: break
    print(json.dumps({"baseline": "daily_rebalanced_equal_weight",
                      "n_assets": env.n_assets,
                      "metrics": metrics_from_infos(metrics, env.initial_cash)}, indent=2))
    env.close()
