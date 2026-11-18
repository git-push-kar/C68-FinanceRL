"""Train only finance modules on the fixed shared InternVL3-2B foundation."""
import argparse
from contextlib import nullcontext
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import random
import subprocess
import time

import numpy as np
import torch

from config import ModelConfig, PPOConfig, serializable
from env import FinanceRewardConfig, PortfolioEnv
from evaluate_policy import baseline_episode, run_episode, tensor_obs
from metrics import metrics_from_infos
from model import InternVLFinancePolicy, digest, write_json
from ppo import compute_gae, ppo_update


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", required=True)
    p.add_argument("--train-end", required=True)
    p.add_argument("--output", required=True, help="New directory only; legacy runs are preserved")
    p.add_argument("--timesteps", type=int, default=10000)
    p.add_argument("--window", type=int, default=30)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--rollout-steps", type=int, default=256)
    p.add_argument("--minibatch-size", type=int, default=16)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--score-scale", type=float, default=1.0)
    p.add_argument("--actor-init-scale", type=float, default=1.0)
    p.add_argument("--initial-log-std", type=float, default=-1.5)
    p.add_argument("--rebalance-rate", type=float, default=1.0)
    p.add_argument("--evaluate-initial", action="store_true")
    p.add_argument("--reward-mode", choices=["legacy_risk", "relative_log", "relative_mean_variance"], default="legacy_risk")
    p.add_argument("--reward-scale", type=float, default=1.0)
    p.add_argument("--initial-package", help="Initialize weights from a compatible package; starts a fresh optimizer")
    p.add_argument("--cost-bps", type=float, default=10)
    p.add_argument("--max-weight", type=float, default=.30)
    p.add_argument("--checkpoint-interval", type=int, default=5000)
    p.add_argument("--validation-interval", type=int, default=0)
    p.add_argument("--validation-start")
    p.add_argument("--validation-end")
    return p


def validate_args(args):
    for key in ("timesteps", "window", "rollout_steps", "minibatch_size"):
        if getattr(args, key) < 1:
            raise ValueError(f"{key} must be positive")
    if min(args.checkpoint_interval, args.validation_interval) < 0:
        raise ValueError("Intervals must be nonnegative")
    if args.learning_rate <= 0:
        raise ValueError("learning-rate must be positive")
    if args.validation_interval:
        if not args.validation_start or not args.validation_end:
            raise ValueError("Periodic validation requires both validation dates")
        if not args.train_end < args.validation_start <= args.validation_end:
            raise ValueError("Require train_end < validation_start <= validation_end")
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Training output must be new or empty")


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Seeds improve repeatability; do not claim bitwise equality across devices.
    torch.backends.cudnn.benchmark = False


def lora_digest(policy):
    checksum = hashlib.sha256()
    for name, parameter in sorted(policy.shared.named_parameters()):
        if "lora_" in name and ".finance." in name:
            checksum.update(name.encode())
            checksum.update(parameter.detach().float().cpu().contiguous().numpy().tobytes())
    return checksum.hexdigest()


def run(args, shared=None):
    validate_args(args)
    seed_everything(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    cfg = ModelConfig(window=args.window, dtype=args.dtype, score_scale=args.score_scale,
                      actor_init_scale=args.actor_init_scale, initial_log_std=args.initial_log_std)
    if args.initial_package:
        from model import validate_package
        parent = validate_package(args.initial_package, allow_architecture_test=bool(shared and shared.architecture_test))
        cfg = ModelConfig(**json.loads((Path(args.initial_package) / 'model_config.json').read_text()))
        if cfg.window != args.window or cfg.dtype != args.dtype:
            raise ValueError('Warm-start window and precision must match the parent package')
        if parent['run'].get('train_end', '9999') > args.train_end:
            raise ValueError('Parent training cutoff exceeds this run cutoff')
    ppo = PPOConfig(learning_rate=args.learning_rate, rollout_steps=args.rollout_steps,
                    minibatch_size=args.minibatch_size)
    reward_cfg = FinanceRewardConfig(transaction_cost_bps=args.cost_bps, max_weight=args.max_weight,
                                     rebalance_rate=args.rebalance_rate, reward_mode=args.reward_mode,
                                     reward_scale=args.reward_scale)
    env = PortfolioEnv(args.data, window=args.window, reward_config=reward_cfg, end_date=args.train_end)
    val_env = None
    if args.validation_interval:
        val_env = PortfolioEnv(args.data, window=args.window, reward_config=reward_cfg,
                               start_date=args.validation_start, end_date=args.validation_end)
        if val_env.tickers != env.tickers:
            raise ValueError("Validation ticker order differs")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if args.initial_package:
        if parent['tickers'] != env.tickers:
            raise ValueError('Warm-start ticker order differs')
        policy = InternVLFinancePolicy.load_artifacts(args.initial_package, shared=shared, device=device,
                    for_training=True, allow_architecture_test=bool(shared and shared.architecture_test))
    else:
        policy = InternVLFinancePolicy(cfg, env.tickers, shared=shared, device=device)
    policy.train()
    params = [p for p in policy.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=ppo.learning_rate)
    use_amp = device.type == "cuda" and args.dtype != "fp32"
    amp_dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    scaler = torch.amp.GradScaler("cuda") if use_amp and args.dtype == "fp16" else None
    try:
        git_head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
        git_dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True, stderr=subprocess.DEVNULL).strip())
    except (OSError, subprocess.CalledProcessError):
        git_head, git_dirty = None, True
    source_hashes = {p.name: digest(p) for p in Path(__file__).parent.glob("*.py")}
    run_meta = {"algorithm": "PPO", "train_end": args.train_end, "seed": args.seed, "command_args": vars(args),
                "base_source_identity": getattr(policy.shared, "source_identity", None),
                "dataset_sha256": digest(args.data), "source_sha256": source_hashes,
                "git_head": git_head, "git_dirty": git_dirty,
                "effective_first_date": str(env.dates[0])[:10], "effective_last_date": str(env.dates[-1])[:10],
                "validation_role": "development_only", "package_versions": {
                    name: importlib.metadata.version(name) for name in ("torch", "transformers", "peft", "gymnasium", "numpy", "pandas")}}
    if args.initial_package:
        run_meta.update(parent_manifest_sha256=digest(Path(args.initial_package) / 'manifest.json'),
                        parent_algorithm=parent['run'].get('algorithm', 'unspecified'))
    write_json(output / "run.json", run_meta)
    initial_lora = lora_digest(policy)
    if args.evaluate_initial:
        if val_env is None:
            raise ValueError("Initial evaluation requires a validation interval and dates")
        write_json(output / "initial_validation.json", {
            "role": "untrained_initial_policy_development_comparison",
            "learned": metrics_from_infos(run_episode(policy, val_env, device, use_amp, amp_dtype)),
            "equal_weight": metrics_from_infos(baseline_episode(val_env))})
    run_meta.update(initial_lora_sha256=initial_lora, optimizer_updates=0, learning_verified=False)
    next_checkpoint = args.checkpoint_interval or float("inf")
    next_validation = args.validation_interval or float("inf")
    obs, _ = env.reset(seed=args.seed)
    steps, episodes = 0, 0
    start = time.monotonic()
    with (output / "training.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = ["steps", "episodes", "seconds", "mean_reward", "mean_net_return", "mean_turnover",
                  "mean_benchmark_net_return", "mean_legacy_drawdown_penalty", "mean_legacy_volatility_penalty", "mean_legacy_turnover_penalty",
                  "loss", "value_loss", "entropy", "approx_kl", "grad_norm", "updates", "early_stop_kl"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        while steps < args.timesteps:
            observations = {k: [] for k in obs}
            actions, logps, values, rewards, dones, infos = [], [], [], [], [], []
            for _ in range(min(ppo.rollout_steps, args.timesteps - steps)):
                with torch.no_grad(), (torch.autocast("cuda", dtype=amp_dtype) if use_amp else nullcontext()):
                    action, logp, value = policy.act(tensor_obs(obs, device))
                for key in obs:
                    observations[key].append(obs[key].copy())
                actions.append(action.squeeze(0).detach().float().clone())
                logps.append(logp.squeeze(0).detach().float())
                values.append(value.squeeze(0).detach().float())
                obs, reward, done, truncated, info = env.step(action[0].float().cpu().numpy().copy())
                rewards.append(reward); dones.append(float(done or truncated)); infos.append(info)
                steps += 1
                if done or truncated:
                    episodes += 1
                    obs, _ = env.reset()
            with torch.no_grad(), (torch.autocast("cuda", dtype=amp_dtype) if use_amp else nullcontext()):
                _, bootstrap = policy(tensor_obs(obs, device))
            advantages, returns = compute_gae(torch.tensor(rewards, device=device),
                torch.stack(values + [bootstrap.squeeze(0).float()]), torch.tensor(dones, device=device), ppo.gamma, ppo.gae_lambda)
            batch = {"obs": {k: torch.as_tensor(np.stack(v), device=device) for k, v in observations.items()},
                     "actions": torch.stack(actions), "log_probs": torch.stack(logps),
                     "advantages": advantages, "returns": returns}
            update = ppo_update(policy, optimizer, batch, ppo, scaler, use_amp, amp_dtype)
            if update.get("updates", 0) < 1:
                raise RuntimeError("PPO performed no optimizer updates in this rollout; aborting training")
            run_meta["optimizer_updates"] += update["updates"]
            row = {"steps": steps, "episodes": episodes, "seconds": time.monotonic() - start,
                   "mean_reward": float(np.mean(rewards)), "mean_net_return": float(np.mean([i["net_return"] for i in infos])),
                   "mean_turnover": float(np.mean([i["turnover"] for i in infos])), **update}
            row.update(mean_benchmark_net_return=float(np.mean([i['benchmark_net_return'] for i in infos])),
                       mean_legacy_drawdown_penalty=float(np.mean([i['legacy_drawdown_penalty'] for i in infos])),
                       mean_legacy_volatility_penalty=float(np.mean([i['legacy_volatility_penalty'] for i in infos])),
                       mean_legacy_turnover_penalty=float(np.mean([i['legacy_turnover_penalty'] for i in infos])))
            writer.writerow(row); stream.flush()
            print(json.dumps(row), flush=True)
            run_meta.update(steps=steps, episodes=episodes, elapsed_seconds=time.monotonic() - start)
            current_lora = lora_digest(policy)
            run_meta.update(final_lora_sha256=current_lora,
                            learning_verified=run_meta["optimizer_updates"] > 0 and current_lora != initial_lora)
            if not run_meta["learning_verified"]:
                raise RuntimeError("Optimizer steps did not change the finance LoRA; no checkpoint will be promoted")
            if steps >= next_checkpoint:
                policy.save_artifacts(output / f"checkpoint_{steps}", serializable(ppo), serializable(reward_cfg), run_meta.copy())
                while next_checkpoint <= steps:
                    next_checkpoint += args.checkpoint_interval
            if steps >= next_validation:
                learned = metrics_from_infos(run_episode(policy, val_env, device, use_amp, amp_dtype))
                baseline = metrics_from_infos(baseline_episode(val_env))
                write_json(output / f"validation_{steps}.json", {"role": "validation", "learned": learned, "equal_weight": baseline})
                while next_validation <= steps:
                    next_validation += args.validation_interval
    policy.save_artifacts(output / "final", serializable(ppo), serializable(reward_cfg), run_meta)
    write_json(output / "learning_check.json", {key: run_meta[key] for key in
               ("optimizer_updates", "initial_lora_sha256", "final_lora_sha256", "learning_verified", "steps")})
    env.close()
    if val_env is not None:
        val_env.close()
    print(f"Saved finance adapter package to {output / 'final'}")
    return policy


if __name__ == "__main__":
    run(build_parser().parse_args())
