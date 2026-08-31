"""Standalone historical Finance LoRA training script; never connects to a broker."""
import argparse
import numpy as np
import torch
from config import ModelConfig, PPOConfig, serializable
from env import FinanceRewardConfig, PortfolioEnv
from model import InternLMFinancePolicy
from ppo import compute_gae, ppo_update


def tensor_obs(obs, device): return {k: torch.as_tensor(v, device=device).unsqueeze(0) for k, v in obs.items()}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--data", required=True); parser.add_argument("--timesteps", type=int, default=25_000)
    parser.add_argument("--window", type=int, default=30); parser.add_argument("--output", default="artifacts/finance")
    parser.add_argument("--backbone", default=ModelConfig.backbone_id); parser.add_argument("--cost-bps", type=float, default=10.0)
    parser.add_argument("--max-weight", type=float, default=0.30); parser.add_argument("--train-end"); parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(); env_cfg = FinanceRewardConfig(transaction_cost_bps=args.cost_bps, max_weight=args.max_weight)
    env = PortfolioEnv(args.data, window=args.window, reward_config=env_cfg, end_date=args.train_end); cfg, ppo_cfg = ModelConfig(backbone_id=args.backbone), PPOConfig(); device = torch.device(args.device)
    policy = InternLMFinancePolicy(cfg, env.n_assets).to(device); optimizer = torch.optim.Adam([p for p in policy.parameters() if p.requires_grad], lr=ppo_cfg.learning_rate)
    obs, _ = env.reset(); steps, rewards = 0, []
    while steps < args.timesteps:
        batch = {"obs": {k: [] for k in obs}, "actions": [], "log_probs": [], "rewards": [], "dones": [], "values": []}
        for _ in range(ppo_cfg.rollout_steps):
            with torch.no_grad(): action, logp, value = policy.act(tensor_obs(obs, device))
            next_obs, reward, terminated, truncated, _ = env.step(action.squeeze(0).cpu().numpy())
            for k, v in obs.items(): batch["obs"][k].append(v)
            batch["actions"].append(action.squeeze(0)); batch["log_probs"].append(logp.squeeze(0)); batch["rewards"].append(torch.tensor(reward, device=device)); batch["dones"].append(torch.tensor(float(terminated or truncated), device=device)); batch["values"].append(value.squeeze(0)); rewards.append(reward); obs = next_obs; steps += 1
            if terminated or truncated: obs, _ = env.reset()
            if steps >= args.timesteps: break
        with torch.no_grad(): _, bootstrap = policy(tensor_obs(obs, device))
        values = torch.stack(batch["values"] + [bootstrap.squeeze(0)]); advantages, returns = compute_gae(torch.stack(batch["rewards"]), values, torch.stack(batch["dones"]), ppo_cfg.gamma, ppo_cfg.gae_lambda)
        batch["obs"] = {k:torch.as_tensor(np.stack(v), device=device) for k,v in batch["obs"].items()}; batch.update(actions=torch.stack(batch["actions"]), log_probs=torch.stack(batch["log_probs"]), advantages=advantages, returns=returns)
        ppo_update(policy, optimizer, batch, ppo_cfg); print(f"steps={steps} mean_reward={np.mean(rewards[-len(batch['rewards']):]):.6f}")
    env.close(); policy.save_artifacts(args.output, serializable(cfg), serializable(ppo_cfg), env_cfg.__dict__, env.tickers)
