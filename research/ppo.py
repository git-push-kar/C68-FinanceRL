"""On-policy PPO with float32 likelihoods, finite checks and KL stopping."""
from contextlib import nullcontext
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


def ppo_update(policy, optimizer, batch, cfg, scaler=None, use_amp=False, amp_dtype=torch.bfloat16):
    advantages = batch["advantages"].float()
    advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
    records = []
    for _ in range(cfg.update_epochs):
        for indices in torch.randperm(len(advantages), device=advantages.device).split(cfg.minibatch_size):
            obs = {key: value[indices] for key, value in batch["obs"].items()}
            context = torch.autocast("cuda", dtype=amp_dtype) if use_amp else nullcontext()
            with context:
                new_logp, values, entropy = policy.evaluate_actions(obs, batch["actions"][indices])
            log_ratio = new_logp.float() - batch["log_probs"][indices].float()
            ratio = log_ratio.exp()
            approx_kl = ((ratio - 1) - log_ratio).mean()
            if not torch.isfinite(approx_kl):
                raise FloatingPointError("Nonfinite PPO likelihood ratio")
            if cfg.target_kl and approx_kl.detach().item() > cfg.target_kl:
                if not records:
                    raise RuntimeError(
                        f"PPO made ZERO updates: initial KL={approx_kl.detach().item():.6g}, "
                        f"max log-prob difference={log_ratio.detach().abs().max().item():.6g}. "
                        "Rollout and learner likelihoods disagree before any optimizer step. "
                        "Run likelihood_check.py; do not treat this as completed training."
                    )
                return {"updates": len(records), "early_stop_kl": float(approx_kl.detach()),
                        **(records[-1] if records else {})}
            surrogate = torch.minimum(ratio * advantages[indices],
                                      ratio.clamp(1 - cfg.clip_range, 1 + cfg.clip_range) * advantages[indices])
            value_loss = nn.functional.mse_loss(values.float(), batch["returns"][indices].float())
            loss = -surrogate.mean() + cfg.value_coef * value_loss - cfg.entropy_coef * entropy.float().mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite PPO loss; aborting without saving a promoted package")
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
            else:
                loss.backward()
            norm = nn.utils.clip_grad_norm_([p for p in policy.parameters() if p.requires_grad],
                                           cfg.max_grad_norm, error_if_nonfinite=True)
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            records.append({"loss": float(loss.detach()), "value_loss": float(value_loss.detach()),
                            "entropy": float(entropy.detach().mean()), "approx_kl": float(approx_kl.detach()),
                            "grad_norm": float(norm)})
    return {"updates": len(records), **records[-1]}
