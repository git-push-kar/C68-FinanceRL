"""Conclusive State-Aware Financial Reinforcement Learning on the Full Nifty Universe.

Operates across all 49 constituent equities with full portfolio state awareness
(cash, holdings, weights, affordabilities, equity, technicals, ranks).
Features whole-share execution, fee and slippage modeling, turnover deadband,
and cross-asset attention policy network.
"""

import argparse
import copy
import datetime
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from research.counterfactual_rl.market import Market, execute, project_scores, baseline

SEEDS = (42, 123, 456)
CAPITALS = [500, 5000, 50000, 100000, 200000]
HORIZON = 20
DEADBAND = 0.015  # 1.5% turnover deadband to prevent wasteful micro-rebalancing fees


def extract_state_matrix(obs):
    """Extract a 49 x 25 state matrix from observation dictionary."""
    stocks = obs['stocks']
    eq = max(float(obs['equity']), 1.0)
    cash = max(float(obs['cash']), 0.0)
    init_cap = max(float(obs['initial_capital']), 1.0)
    n = len(stocks)

    ranks = {
        k: np.argsort(np.argsort([r[k] for r in stocks], kind='stable'), kind='stable') / max(n - 1, 1)
        for k in ['return20', 'return60', 'return120', 'vol20']
    }

    feat = []
    for i, r in enumerate(stocks):
        q = float(r['historical_quote'])
        shares = int(r['held_shares'])
        pos_val = float(r['position_value'])
        afford = int(r['affordable_shares_cash'])

        row = [
            np.log1p(q) / 12.0,
            np.log1p(shares) / 12.0,
            pos_val / eq,
            float(r['weight']),
            afford * q / eq,
            float(r['return5']) / 0.1,
            float(r['return20']) / 0.2,
            float(r['return60']) / 0.4,
            float(r['return120']) / 0.6,
            float(r['vol20']) / 0.4,
            float(r['drawdown60']) / 0.4,
            float(np.log(max(r['volume_ratio'], 0.01))),
            np.log1p(init_cap) / 12.0,
            np.log1p(cash) / 12.0,
            np.log1p(eq) / 12.0,
            cash / eq,
            float(obs['max_stock_weight']),
            float(obs['fees_bps']) / 100.0,
            float(obs['slippage_bps']) / 100.0,
            float(shares > 0),
            float(ranks['return20'][i]),
            float(ranks['return60'][i]),
            float(ranks['return120'][i]),
            float(ranks['vol20'][i]),
            q / eq,
        ]
        feat.append(row)
    return np.asarray(feat, dtype=np.float32)


class CrossAssetAttentionPolicy(nn.Module):
    """Deep Cross-Asset Attention Policy Network.
    
    Observes all 49 stocks simultaneously. Uses multi-head self-attention
    to model cross-asset interactions, relative valuation, and portfolio state.
    """
    def __init__(self, in_features=25, hidden_dim=128, num_heads=4, num_layers=2):
        super().__init__()
        self.in_features = in_features
        self.hidden_dim = hidden_dim

        self.feature_encoder = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=0.0,
            activation='gelu',
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.global_context = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
        )

        self.score_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        self.gate_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        self.cash_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x):
        """Forward pass.
        
        Args:
            x: Tensor of shape (batch, 49, 25)
        Returns:
            scores: Tensor of shape (batch, 49) non-negative preferences
            gates: Tensor of shape (batch, 49) in [0, 1]
            cash_pref: Tensor of shape (batch, 1) non-negative cash preference
        """
        batch_size = x.size(0)
        h = self.feature_encoder(x)  # (batch, 49, hidden)
        z = self.transformer(h)      # (batch, 49, hidden)

        # Global market & portfolio context
        z_mean = z.mean(dim=1)
        z_max = z.max(dim=1).values
        g = self.global_context(torch.cat([z_mean, z_max], dim=-1))  # (batch, hidden)
        g_expanded = g.unsqueeze(1).expand(-1, 49, -1)               # (batch, 49, hidden)

        combined = torch.cat([z, g_expanded], dim=-1)  # (batch, 49, hidden * 2)

        raw_scores = self.score_head(combined).squeeze(-1)  # (batch, 49)
        scores = F.softplus(raw_scores)

        raw_gates = self.gate_head(combined).squeeze(-1)    # (batch, 49)
        gates = torch.sigmoid(raw_gates)

        cash_pref = F.softplus(self.cash_head(z_mean)).squeeze(-1)  # (batch,)

        return scores, gates, cash_pref

    def compute_weights(self, obs, deadband=DEADBAND, explore_std=0.0):
        """Compute portfolio target weights for an observation."""
        state = extract_state_matrix(obs)
        tensor_state = torch.from_numpy(state).unsqueeze(0).to(
            next(self.parameters()).device
        )
        with torch.no_grad():
            scores, gates, cash_pref = self(tensor_state)
            s = scores[0].cpu().numpy()
            g = gates[0].cpu().numpy()
            c = cash_pref[0].cpu().item()

        if explore_std > 0.0:
            noise = np.random.normal(0, explore_std, size=s.shape)
            s = np.maximum(0.0, s + noise)

        # Combine score with gate
        active_scores = s * (g > 0.3)

        # Apply cash dampening if cash preference is elevated
        if c > 2.0:
            active_scores *= (2.0 / c)

        # Ensure at least some active scores or fallback to equal weight
        if active_scores.sum() < 1e-6:
            target_weights = np.ones(49) / 49.0
        else:
            target_weights = project_scores(active_scores)

        # Turnover deadband: If current weight is already close, don't trade
        if deadband > 0.0:
            curr_weights = np.array([r['weight'] for r in obs['stocks']], dtype=np.float64)
            diff = np.abs(target_weights - curr_weights)
            keep_mask = (diff < deadband) & (curr_weights > 0.0)
            target_weights[keep_mask] = curr_weights[keep_mask]
            # Renormalize if sum exceeds 1
            if target_weights.sum() > 1.0:
                target_weights /= target_weights.sum()

        return target_weights


def simulate_policy(m, policy, start, end, capital, deadband=DEADBAND):
    """Simulate policy execution over specified period."""
    t, last = m.bounds(start, end)
    cash = float(capital)
    qty = np.zeros(49, np.int64)
    pending = np.zeros(2)
    daily_equity = []
    decisions = []
    total_fees = 0.0
    total_slippage = 0.0

    while t < last:
        obs = m.observe(t, cash, qty, capital, pending)
        weights = policy.compute_weights(obs, deadband=deadband)
        next_t = min(t + HORIZON, last)
        res = m.horizon(t, cash, qty, weights, next_t, pending)

        decisions.append({
            'signal_date': str(m.dates[t]),
            'execution_date': str(m.dates[t + 1]),
            'weights': weights.tolist(),
            'trade_shares': res['trade_shares'].tolist(),
            'cash': float(res['cash']),
            'equity': float(res['values'][-1]),
            'fees': float(res['fees']),
            'slippage': float(res['slippage']),
        })

        cash, qty, pending = res['cash'], res['qty'], res['pending']
        daily_equity.extend(res['values'].tolist())
        total_fees += res['fees']
        total_slippage += res['slippage']
        t = next_t

    v = np.r_[capital, daily_equity]
    rets = v[1:] / v[:-1] - 1.0
    mean_ret = rets.mean()
    std_ret = max(rets.std(), 1e-12)
    sharpe = float(np.sqrt(252) * mean_ret / std_ret)
    drawdowns = 1.0 - v / np.maximum.accumulate(v)
    max_dd = float(drawdowns.max())
    net_return = float(v[-1] / capital - 1.0)

    # Sortino ratio (downside deviation)
    neg_rets = rets[rets < 0.0]
    downside_std = max(neg_rets.std() if len(neg_rets) > 0 else std_ret, 1e-12)
    sortino = float(np.sqrt(252) * mean_ret / downside_std)

    metrics = {
        'return': net_return,
        'end_equity': float(v[-1]),
        'sharpe': sharpe,
        'sortino': sortino,
        'max_drawdown': max_dd,
        'fees': total_fees,
        'slippage': total_slippage,
        'sessions': len(rets),
        'decisions_count': len(decisions),
    }

    return {
        'metrics': metrics,
        'daily_equity': daily_equity,
        'decisions': decisions,
    }


def train_policy_seed(seed, m, device, total_epochs=60, batch_trajs=16):
    """Train state-aware cross-asset attention policy on 2018-2022."""
    print(f"\n==========================================")
    print(f"Starting Training for Seed {seed} on {device}")
    print(f"==========================================")

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    policy = CrossAssetAttentionPolicy().to(device)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs, eta_min=1e-5)

    first, last = m.bounds('2018-07-02', '2022-12-31')
    val_years = (2023, 2024)

    best_val_score = -float('inf')
    best_checkpoint_weights = None
    best_val_metrics = None
    history = []

    for epoch in range(1, total_epochs + 1):
        policy.train()
        trajectories = []
        explore_std = max(0.05, 0.3 * (1.0 - epoch / total_epochs))

        # Sample batch of multi-period trajectories
        log_probs = []
        rewards = []

        for b in range(batch_trajs):
            t_start = int(np.random.randint(first, last - 80))
            cap = float(np.random.choice([50000, 100000, 200000]))
            steps = int(np.random.randint(3, 5))

            c = cap
            q = np.zeros(49, np.int64)
            p = np.zeros(2)
            t = t_start
            traj_values = [cap]
            traj_fees = 0.0

            step_log_probs = []

            for step in range(steps):
                obs = m.observe(t, c, q, cap, p)
                state = extract_state_matrix(obs)
                tensor_state = torch.from_numpy(state).unsqueeze(0).to(device)

                scores, gates, cash_pref = policy(tensor_state)

                # Continuous score policy with exploration noise
                s = scores[0]
                noise = torch.randn_like(s) * explore_std
                noisy_scores = F.relu(s + noise)

                # Log probability under Gaussian exploration
                dist = torch.distributions.Normal(s, explore_std)
                step_log_prob = dist.log_prob(noisy_scores).sum()
                step_log_probs.append(step_log_prob)

                # Compute execution weights
                active_s = (noisy_scores * (gates[0] > 0.3)).detach().cpu().numpy()
                if active_s.sum() < 1e-6:
                    w = np.ones(49) / 49.0
                else:
                    w = project_scores(active_s)

                # Turnover deadband
                curr_w = np.array([r['weight'] for r in obs['stocks']], dtype=np.float64)
                keep_mask = (np.abs(w - curr_w) < DEADBAND) & (curr_w > 0.0)
                w[keep_mask] = curr_w[keep_mask]
                if w.sum() > 1.0:
                    w /= w.sum()

                next_t = min(t + HORIZON, last)
                res = m.horizon(t, c, q, w, next_t, p)
                c, q, p = res['cash'], res['qty'], res['pending']
                traj_values.extend(res['values'].tolist())
                traj_fees += res['fees']
                t = next_t

            # Trajectory return & risk metric
            v = np.array(traj_values)
            tot_ret = float(v[-1] / v[0] - 1.0)
            dd = float((1.0 - v / np.maximum.accumulate(v)).max())
            fee_drag = traj_fees / v[0]

            # Objective: Net growth - drawdown penalty - fee penalty
            reward = tot_ret - 0.5 * (dd ** 2) - 0.5 * fee_drag
            rewards.append(reward)
            log_probs.append(torch.stack(step_log_probs).mean())

        # Policy gradient update with leave-one-out baseline
        rewards_arr = np.array(rewards)
        baseline_rewards = (rewards_arr.sum() - rewards_arr) / max(len(rewards_arr) - 1, 1)
        advantages = torch.from_numpy(rewards_arr - baseline_rewards).float().to(device)

        # Standardize advantages
        if advantages.std() > 1e-6:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        loss = 0.0
        for lp, adv in zip(log_probs, advantages):
            loss -= lp * adv
        loss = loss / batch_trajs

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        # Validation every 5 epochs
        if epoch % 5 == 0 or epoch == 1 or epoch == total_epochs:
            policy.eval()
            val_returns = {}
            for yr in val_years:
                res = simulate_policy(m, policy, f"{yr}-01-01", f"{yr}-12-31", 100000)
                val_returns[yr] = res['metrics']

            score = 0.5 * (np.log1p(val_returns[2023]['return']) + np.log1p(val_returns[2024]['return']))

            if score > best_val_score:
                best_val_score = score
                best_checkpoint_weights = copy.deepcopy(policy.state_dict())
                best_val_metrics = val_returns

            info = {
                'epoch': epoch,
                'loss': float(loss.item()),
                'mean_reward': float(rewards_arr.mean()),
                'val_score': float(score),
                'val_2023_ret': float(val_returns[2023]['return']),
                'val_2024_ret': float(val_returns[2024]['return']),
            }
            history.append(info)
            print(f"Epoch {epoch:2d}/{total_epochs} | Loss: {loss.item():6.4f} | "
                  f"Val 2023: {val_returns[2023]['return']*100:5.2f}% | "
                  f"Val 2024: {val_returns[2024]['return']*100:5.2f}% | "
                  f"Best Score: {best_val_score:6.4f}")

    # Load best checkpoint
    policy.load_state_dict(best_checkpoint_weights)
    return policy, history, best_val_metrics


def run_full_evaluation(seed, policy, m, output_dir):
    """Run comprehensive test evaluation across capitals and baselines."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    policy.eval()

    start, end = "2025-01-01", "2026-01-30"
    results = {}

    print(f"\n--- Running Full Test Evaluation for Seed {seed} (2025-01-01 to 2026-01-30) ---")

    # Primary ₹100,000 comparison across all baselines
    primary_sims = {}
    for b_name in ['buy_hold', 'equal_rebalanced', 'inverse_vol', 'momentum', 'cash']:
        res = m.bounds(start, end)
        sim_res = evaluate_baseline_named(m, b_name, start, end, 100000)
        primary_sims[b_name] = sim_res

    # Our RL policy
    rl_sim = simulate_policy(m, policy, start, end, 100000)
    primary_sims['state_aware_rl'] = rl_sim

    # Multi-capital stress testing
    capital_results = {}
    for cap in CAPITALS:
        cap_sims = {}
        # RL
        r = simulate_policy(m, policy, start, end, cap)
        cap_sims['state_aware_rl'] = r['metrics']
        # Buy & Hold
        bh = evaluate_baseline_named(m, 'buy_hold', start, end, cap)
        cap_sims['buy_hold'] = bh['metrics']
        # Equal Rebalanced
        eq = evaluate_baseline_named(m, 'equal_rebalanced', start, end, cap)
        cap_sims['equal_rebalanced'] = eq['metrics']

        capital_results[cap] = cap_sims
        print(f"Capital ₹{cap:7d} -> RL: {r['metrics']['return']*100:6.2f}% | "
              f"Hold: {bh['metrics']['return']*100:6.2f}% | "
              f"Eq: {eq['metrics']['return']*100:6.2f}%")

    # Bootstrap confidence intervals
    rl_equity = primary_sims['state_aware_rl']['daily_equity']
    bootstraps = {}
    for comp in ['buy_hold', 'equal_rebalanced', 'momentum']:
        comp_equity = primary_sims[comp]['daily_equity']
        b_res = compute_block_bootstrap(rl_equity, comp_equity, 100000, seed=seed)
        bootstraps[comp] = b_res
        ci = b_res['block20_percentile95']
        print(f"RL vs {comp:16s} Diff: {b_res['return_diff_pp']:+5.2f} pp | 95% CI: [{ci[0]:+5.2f}, {ci[1]:+5.2f}] pp")

    # Save artifacts
    results = {
        'seed': seed,
        'primary_metrics': {k: v['metrics'] for k, v in primary_sims.items()},
        'capital_stress': capital_results,
        'bootstraps': bootstraps,
        'rl_decisions': primary_sims['state_aware_rl']['decisions'],
        'rl_daily_equity': primary_sims['state_aware_rl']['daily_equity'],
    }

    (output_dir / f"seed{seed}_results.json").write_text(json.dumps(results, indent=2))
    return results


def evaluate_baseline_named(m, kind, start, end, capital):
    """Wrapper to evaluate baseline in market."""
    from research.counterfactual_rl.market import evaluate as m_eval
    return m_eval(m, baseline(kind), start, end, capital)


def compute_block_bootstrap(equity_a, equity_b, initial, n_blocks=2000, block_size=20, seed=42):
    """Compute circular block-bootstrap confidence interval between two daily equity paths."""
    rng = np.random.default_rng(seed)
    a = np.r_[initial, equity_a]
    b = np.r_[initial, equity_b]
    diff = np.log(a[1:] / a[:-1]) - np.log(b[1:] / b[:-1])
    n = len(diff)

    boot_diffs = []
    for _ in range(n_blocks):
        starts = rng.integers(0, n, int(np.ceil(n / block_size)))
        indices = np.concatenate([(s + np.arange(block_size)) % n for s in starts])[:n]
        boot_diffs.append(float(diff[indices].sum()) * 100.0)

    ci95 = np.percentile(boot_diffs, [2.5, 97.5]).tolist()
    return {
        'return_diff_pp': float((a[-1] / initial - b[-1] / initial) * 100.0),
        'log_growth_diff_pp': float(diff.sum() * 100.0),
        'block20_percentile95': ci95,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', default='research/conclusive_rl_runs')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    m = Market()
    all_seed_results = {}

    for seed in SEEDS:
        policy, hist, val_metrics = train_policy_seed(
            seed, m, args.device, total_epochs=args.epochs, batch_trajs=16
        )
        # Save model checkpoint
        torch.save(policy.state_dict(), out_dir / f"model_seed{seed}.pt")
        # Evaluate
        res = run_full_evaluation(seed, policy, m, out_dir)
        res['training_history'] = hist
        res['val_metrics'] = val_metrics
        all_seed_results[seed] = res

    # Generate master summary
    summary_path = out_dir / "conclusive_summary.json"
    summary_data = {
        'timestamp': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'seeds': list(SEEDS),
        'results': {
            s: {
                'primary_metrics': all_seed_results[s]['primary_metrics'],
                'capital_stress': all_seed_results[s]['capital_stress'],
                'bootstraps': all_seed_results[s]['bootstraps'],
            }
            for s in SEEDS
        }
    }
    summary_path.write_text(json.dumps(summary_data, indent=2))
    print(f"\nTraining and Evaluation Complete for All 3 Seeds! Results saved to {summary_path}")


if __name__ == '__main__':
    main()
