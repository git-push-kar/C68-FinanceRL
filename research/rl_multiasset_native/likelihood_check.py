"""GPU gate: compare rollout and learner likelihoods BEFORE any parameter update."""
import argparse
from contextlib import nullcontext
import json

import numpy as np
import torch

from config import ModelConfig
from env import PortfolioEnv
from evaluate_policy import tensor_obs
from model import InternVLFinancePolicy, write_json


def probe_likelihoods(policy, observations, actions, old_log_probs, use_amp=False, amp_dtype=torch.bfloat16):
    results = {}
    for label, native, gradients in [("native_batched_grad", True, True),
                                      ("serial_no_grad", False, False), ("serial_grad", False, True)]:
        with torch.set_grad_enabled(gradients), (torch.autocast("cuda", dtype=amp_dtype) if use_amp else nullcontext()):
            mean, _ = policy._forward_native(observations) if native else policy(observations)
            logp = policy._dist(mean).log_prob(actions).sum(-1)
        delta = (logp.detach().float() - old_log_probs.float())
        kl = ((delta.exp() - 1) - delta).mean()
        results[label] = {"approx_kl": float(kl), "max_abs_logp_difference": float(delta.abs().max())}
        del mean, logp
    results["passed"] = all(
        np.isfinite(results[label]["approx_kl"]) and results[label]["approx_kl"] <= 1e-4
        and results[label]["max_abs_logp_difference"] <= .01
        for label in ("serial_no_grad", "serial_grad")
    )
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/nifty50_prices.csv")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    torch.manual_seed(42)
    device = torch.device(args.device)
    env = PortfolioEnv(args.data, end_date="2022-12-31")
    cfg = ModelConfig(dtype=args.dtype)
    policy = InternVLFinancePolicy(cfg, env.tickers, device=device).train()
    use_amp = device.type == "cuda" and args.dtype == "bf16"
    obs, _ = env.reset()
    observations = {k: [] for k in obs}
    actions, logps = [], []
    for _ in range(args.samples):
        for key in obs:
            observations[key].append(obs[key].copy())
        with torch.no_grad(), (torch.autocast("cuda", dtype=torch.bfloat16) if use_amp else nullcontext()):
            action, logp, _ = policy.act(tensor_obs(obs, device))
        actions.append(action[0].detach().float().clone())
        logps.append(logp[0].detach().float())
        obs, _, done, truncated, _ = env.step(action[0].float().cpu().numpy().copy())
        if done or truncated:
            obs, _ = env.reset()
    batch = {k: torch.as_tensor(np.stack(v), device=device) for k, v in observations.items()}
    result = {"dtype": args.dtype, "samples": args.samples,
              **probe_likelihoods(policy, batch, torch.stack(actions), torch.stack(logps), use_amp)}
    write_json(args.output, result)
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise RuntimeError("Rollout/learner likelihood gate failed before training; see the numerical probe JSON")


if __name__ == "__main__":
    main()
