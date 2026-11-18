"""Train numeric finance tokens/LoRA to rank forward returns before PPO."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from config import ModelConfig, serializable
from env import FinanceRewardConfig, PortfolioEnv
from evaluate_policy import baseline_episode, run_episode
from metrics import metrics_from_infos
from model import InternVLFinancePolicy, digest, write_json
from train import lora_digest, seed_everything


def label_indices(length, window, horizon, split_fraction=.8):
    boundary = int(length * split_fraction)
    train = np.arange(window, boundary - horizon)
    holdout = np.arange(max(window, boundary), length - horizon)
    if len(train) < 2 or len(holdout) < 2:
        raise ValueError('Not enough training history for purged warm-up split')
    assert train[-1] + horizon < boundary <= holdout[0]
    assert holdout[-1] + horizon < length
    return train, holdout, boundary


def rank_targets(forward_returns):
    # Average ranks make ties neutral, including a completely flat market.
    _, inverse, counts = np.unique(forward_returns, return_inverse=True, return_counts=True)
    ends = counts.cumsum() - 1
    starts = ends - counts + 1
    ranks = ((starts + ends) / 2)[inverse]
    return (ranks / (len(ranks) - 1) * 2 - 1).astype(np.float32) * .5


def risk_rule_scores(market):
    volatility = np.maximum(market[-20:, :, 0].astype(np.float64).std(0), 1e-4)
    scores = -np.log(volatility)
    scores = np.clip(scores - scores.mean(), -2, 2)
    return (scores - scores.mean()).astype(np.float32)


def examples(env, indices, horizon, target_kind='future_rank'):
    observations, labels = [], []
    for index in indices:
        env.index = int(index)
        # Use a neutral portfolio so targets cannot depend on future holdings.
        env.weights = np.full(env.n_assets, 1 / env.n_assets)
        env.cash_weight = 0.
        env.equity = env.peak_equity = env.initial_cash
        env.returns = []
        observations.append(env._observation())
        if target_kind == 'risk_rule':
            labels.append(risk_rule_scores(observations[-1]['market']))
        else:
            forward = env.prices[index + horizon] / env.prices[index] - 1
            labels.append(rank_targets(forward))
    return ({k: torch.as_tensor(np.stack([o[k] for o in observations]), dtype=torch.float32)
             for k in observations[0]}, torch.as_tensor(np.stack(labels)))


def predict_probe(policy, probe, device):
    observations, targets = probe
    predictions = []
    with torch.no_grad():
        for i in range(len(targets)):
            scores, _ = policy({k: v[i:i+1].to(device) for k, v in observations.items()})
            scores = scores - scores.mean(-1, keepdim=True)
            predictions.append(scores.cpu())
    prediction = torch.cat(predictions)
    correlations = []
    for actual, target in zip(prediction.numpy(), targets.numpy()):
        ranks = rank_targets(actual)
        if np.std(ranks) > 0 and np.std(target) > 0:
            correlations.append(float(np.corrcoef(ranks, rank_targets(target))[0, 1]))
    return {'rank_target_mse': float((prediction - targets).square().mean()),
            'mean_cross_sectional_rank_correlation': float(np.mean(correlations)) if correlations else None,
            'sampled_dates': len(targets),
            'note': 'Dates may overlap in their forward horizons; this is a diagnostic, not a significance test.'}


def shuffle_assets(observations, targets, generator):
    """Move each asset's history, holdings and target together to a new slot."""
    result = {k: v.clone() for k, v in observations.items()}
    labels = targets.clone()
    assets = targets.shape[1]
    for i in range(len(targets)):
        order = torch.randperm(assets, generator=generator)
        result['market'][i] = observations['market'][i].index_select(1, order)
        result['portfolio'][i, :assets] = observations['portfolio'][i, :assets].index_select(0, order)
        labels[i] = targets[i].index_select(0, order)
    return result, labels


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--train-end', default='2022-12-31')
    p.add_argument('--validation-start', default='2023-01-01')
    p.add_argument('--validation-end', default='2024-12-31')
    p.add_argument('--steps', type=int, default=512)
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--horizon', type=int, default=20)
    p.add_argument('--window', type=int, default=30)
    p.add_argument('--max-weight', type=float, default=.3)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', default='cuda')
    p.add_argument('--feature-driven', action='store_true',
                   help='Disable ticker embeddings, normalize numeric inputs and shuffle asset slots during supervised fitting')
    p.add_argument('--target-kind', choices=['future_rank', 'risk_rule'], default='future_rank')
    return p


def run(args, shared=None):
    if min(args.steps, args.batch_size, args.horizon, args.window) < 1:
        raise ValueError('Step counts, horizon, batch and window must be positive')
    if not args.train_end < args.validation_start <= args.validation_end:
        raise ValueError('Warm-up labels must end before the development period')
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Use a new warm-up output directory')
    output.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)
    device = torch.device(args.device)
    cfg = ModelConfig(window=args.window, dtype='fp32', score_scale=1., actor_init_scale=.01, initial_log_std=-1.5,
                      use_asset_ids=not args.feature_driven, normalize_market=args.feature_driven)
    reward = FinanceRewardConfig(max_weight=args.max_weight, rebalance_rate=.1,
                                 reward_mode='relative_mean_variance' if args.target_kind == 'risk_rule' else 'relative_log', reward_scale=100)
    env = PortfolioEnv(args.data, window=args.window, reward_config=reward, end_date=args.train_end)
    validation = PortfolioEnv(args.data, window=args.window, reward_config=reward,
                               start_date=args.validation_start, end_date=args.validation_end)
    effective_horizon = 0 if args.target_kind == 'risk_rule' else args.horizon
    train_indices, holdout_indices, boundary = label_indices(len(env.prices), args.window, effective_horizon)
    rng = np.random.default_rng(args.seed)
    fixed = np.linspace(0, len(holdout_indices) - 1, min(32, len(holdout_indices)), dtype=int)
    probe = examples(env, holdout_indices[fixed], effective_horizon, args.target_kind)
    fixed_train = np.linspace(0, len(train_indices) - 1, min(32, len(train_indices)), dtype=int)
    train_probe = examples(env, train_indices[fixed_train], effective_horizon, args.target_kind)
    permutation_rng = torch.Generator().manual_seed(args.seed + 1)
    policy = InternVLFinancePolicy(cfg, env.tickers, shared=shared, device=device)
    # Value is not a forward-return rank; PPO will fit it after warm-up.
    with torch.no_grad():
        policy.critic.weight.zero_(); policy.critic.bias.zero_()
    policy.critic.requires_grad_(False)
    policy.log_std.requires_grad_(False)
    lora = [p for n, p in policy.shared.named_parameters() if p.requires_grad and 'lora_' in n]
    optimizer = torch.optim.Adam([{'params': policy.actor.parameters(), 'lr': 1e-4},
                                  {'params': policy.encoders.parameters(), 'lr': 1e-5},
                                  {'params': lora, 'lr': 1e-5}])
    metadata = {'algorithm': 'supervised_numeric_warmup_not_PPO', 'train_end': args.train_end,
                'target_kind': args.target_kind, 'horizon_trading_days': effective_horizon, 'seed': args.seed, 'command_args': vars(args),
                'dataset_sha256': digest(args.data), 'base_source_identity': getattr(policy.shared, 'source_identity', None),
                'supervised_label_last_date': str(env.dates[train_indices[-1] + effective_horizon])[:10],
                'internal_holdout_first_feature_date': str(env.dates[boundary])[:10],
                'validation_role': 'development_only', 'initial_lora_sha256': lora_digest(policy),
                'target': 'causal inverse-volatility risk rule, not a future-return forecast' if args.target_kind == 'risk_rule' else 'cross-sectional average ranks of forward returns, scaled to [-0.5, 0.5]',
                'feature_driven': args.feature_driven,
                'training_augmentation': 'asset histories, holdings and targets permuted together per example' if args.feature_driven else 'none',
                'source_sha256': {p.name: digest(p) for p in Path(__file__).parent.glob('*.py')}}
    write_json(output / 'run.json', metadata)
    initial = {'role': 'initial_warmup_policy', 'learned': metrics_from_infos(run_episode(policy, validation, device)),
               'equal_weight': metrics_from_infos(baseline_episode(validation)),
               'internal_holdout_probe': predict_probe(policy, probe, device),
               'fixed_training_probe': predict_probe(policy, train_probe, device)}
    write_json(output / 'initial_validation.json', initial)
    initial_metadata = dict(metadata, algorithm='untrained_initialization', steps=0, optimizer_updates=0, learning_verified=False, labels_consumed=0)
    policy.save_artifacts(output / 'initial', {'algorithm': 'untrained_initialization'}, serializable(reward), initial_metadata)
    logs = []
    interval = max(1, args.steps // 4)
    for step in range(1, args.steps + 1):
        observations, target = examples(env, rng.choice(train_indices, size=args.batch_size), effective_horizon, args.target_kind)
        if args.feature_driven:
            observations, target = shuffle_assets(observations, target, permutation_rng)
        scores, _ = policy({k: v.to(device) for k, v in observations.items()})
        centered = scores - scores.mean(-1, keepdim=True)
        loss = (centered - target.to(device)).square().mean()
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite warm-up loss')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_([p for p in policy.parameters() if p.requires_grad], 1., error_if_nonfinite=True)
        optimizer.step()
        if step == 1 or step % 32 == 0:
            row = {'steps': step, 'rank_target_mse': float(loss.detach()), 'grad_norm_before_clip': float(norm)}
            logs.append(row); write_json(output / 'losses.json', logs); print(json.dumps(row), flush=True)
        if step % interval == 0 or step == args.steps:
            current = lora_digest(policy)
            metadata.update(steps=step, optimizer_updates=step, final_lora_sha256=current,
                            learning_verified=current != metadata['initial_lora_sha256'])
            if not metadata['learning_verified']:
                raise RuntimeError('Warm-up did not change LoRA weights')
            policy.save_artifacts(output / f'checkpoint_{step}', {'algorithm': 'supervised_warmup'}, serializable(reward), metadata.copy())
            write_json(output / f'validation_{step}.json', {
                'role': 'development_only', 'learned': metrics_from_infos(run_episode(policy, validation, device)),
                'equal_weight': metrics_from_infos(baseline_episode(validation)),
                'internal_holdout_probe': predict_probe(policy, probe, device),
                'fixed_training_probe': predict_probe(policy, train_probe, device)})
    write_json(output / 'learning_check.json', metadata)
    env.close(); validation.close()
    return policy


if __name__ == '__main__':
    run(parser().parse_args())
