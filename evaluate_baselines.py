"""Standalone equal-weight baseline for the untouched finance test period."""
import argparse
import numpy as np
from env import PortfolioEnv

if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--data", required=True); parser.add_argument("--start", required=True); parser.add_argument("--end", required=True); parser.add_argument("--window", type=int, default=30); args = parser.parse_args()
    env = PortfolioEnv(args.data, window=args.window, start_date=args.start, end_date=args.end); obs, _ = env.reset(); metrics = []
    while True:
        obs, _, terminated, truncated, info = env.step(np.zeros(env.n_assets, dtype=np.float32)); metrics.append(info)
        if terminated or truncated: break
    print({"baseline":"equal_weight", "final_equity":metrics[-1]["equity"], "mean_net_return":float(np.mean([m["net_return"] for m in metrics])), "max_drawdown":max(m["drawdown"] for m in metrics)})
    env.close()
