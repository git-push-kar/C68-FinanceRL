"""Evaluate a versioned InternVL3 finance package on an explicit date block."""
import argparse
from contextlib import nullcontext
import json
from pathlib import Path

import numpy as np
import torch

from env import FinanceRewardConfig, PortfolioEnv
from metrics import metrics_from_infos
from model import InternVLFinancePolicy, validate_package, write_json


def tensor_obs(obs, device):
    return {k: torch.as_tensor(v, device=device).unsqueeze(0) for k, v in obs.items()}


@torch.no_grad()
def run_episode(policy, env, device, use_amp=None, amp_dtype=torch.bfloat16, deterministic=True):
    if use_amp is None:
        use_amp = torch.device(device).type == "cuda" and policy.cfg.dtype != "fp32"
        amp_dtype = torch.float16 if policy.cfg.dtype == "fp16" else torch.bfloat16
    was_training = policy.training
    policy.eval()
    try:
        obs, _ = env.reset()
        infos = []
        while True:
            with torch.autocast("cuda", dtype=amp_dtype) if use_amp else nullcontext():
                if deterministic:
                    action, _ = policy(tensor_obs(obs, device))
                else:
                    action, _, _ = policy.act(tensor_obs(obs, device))
            obs, _, terminated, truncated, info = env.step(action.squeeze(0).float().cpu().numpy().copy())
            infos.append(info)
            if terminated or truncated:
                return infos
    finally:
        policy.train(was_training)


def baseline_episode(env):
    # Keep the benchmark daily-rebalanced even for a slower-trading learned policy.
    rate = env.reward_config.rebalance_rate
    env.reward_config.rebalance_rate = 1.0
    try:
        return _daily_equal_weight_episode(env)
    finally:
        env.reward_config.rebalance_rate = rate


def _daily_equal_weight_episode(env):
    env.reset()
    infos = []
    while True:
        _, _, done, truncated, info = env.step(np.zeros(env.n_assets))
        infos.append(info)
        if done or truncated:
            return infos


def load_policy(artifact_dir, device):
    policy = InternVLFinancePolicy.load_artifacts(artifact_dir, device=device)
    return policy, len(policy.tickers)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--artifacts", required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--output", required=True, help="Fresh evaluation JSON; never overwrites historical results")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--cost-bps", type=float)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError("Choose a new evaluation output file")
    path = Path(args.artifacts)
    manifest = validate_package(path)
    if args.start > args.end:
        parser.error("start must be <= end")
    if manifest["run"].get("train_end", "9999") >= args.start:
        raise ValueError("Evaluation overlaps or lacks a recorded training cutoff")
    env_config = json.loads((path / "environment_config.json").read_text())
    reward = FinanceRewardConfig(**env_config["reward"])
    if args.cost_bps is not None:
        reward.transaction_cost_bps = args.cost_bps
    env = PortfolioEnv(args.data, window=manifest["window"], reward_config=reward,
                       start_date=args.start, end_date=args.end)
    if env.tickers != manifest["tickers"]:
        raise ValueError("Dataset ticker order differs from the trained package")
    policy, _ = load_policy(path, torch.device(args.device))
    learned = metrics_from_infos(run_episode(policy, env, args.device), env.initial_cash)
    baseline = metrics_from_infos(baseline_episode(env), env.initial_cash)
    env.close()
    result = {"requested_start": args.start, "requested_end": args.end,
              "base_model": manifest["base_model_id"], "base_revision": manifest["base_revision"],
              "cost_bps": reward.transaction_cost_bps,
              "learned_policy": learned, "daily_rebalanced_equal_weight": baseline,
              "excess_return_percentage_points": 100 * (learned["return_total"] - baseline["return_total"]),
              "relative_final_wealth": learned["final_equity"] / baseline["final_equity"] - 1}
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
