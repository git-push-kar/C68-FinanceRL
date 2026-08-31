"""On-policy PPO: GAE, clipped objective, value loss, and entropy bonus."""
import torch
from torch import nn


def compute_gae(rewards, values, dones, gamma, gae_lambda):
    advantages = torch.zeros_like(rewards)
    advantage = torch.zeros((), device=rewards.device)
    for t in reversed(range(len(rewards))):
        nonterminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * values[t + 1] * nonterminal - values[t]
        advantage = delta + gamma * gae_lambda * nonterminal * advantage
        advantages[t] = advantage
    return advantages, advantages + values[:-1]


def ppo_update(policy: nn.Module, optimizer, batch, cfg):
    advantages = (batch["advantages"] - batch["advantages"].mean()) / (batch["advantages"].std() + 1e-8)
    for _ in range(cfg.update_epochs):
        for indices in torch.randperm(len(advantages), device=advantages.device).split(cfg.minibatch_size):
            obs = {key: value[indices] for key, value in batch["obs"].items()}
            new_logp, values, entropy = policy.evaluate_actions(obs, batch["actions"][indices])
            ratio = (new_logp - batch["log_probs"][indices]).exp()
            surrogate = torch.minimum(ratio * advantages[indices],
                                      ratio.clamp(1 - cfg.clip_range, 1 + cfg.clip_range) * advantages[indices])
            loss = (-surrogate.mean() + cfg.value_coef * nn.functional.mse_loss(values, batch["returns"][indices])
                    - cfg.entropy_coef * entropy.mean())
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), cfg.max_grad_norm)
            optimizer.step()
