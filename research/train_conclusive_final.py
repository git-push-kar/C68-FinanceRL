"""Refined Conclusive State-Aware Financial Reinforcement Learning on Full 49-Stock Universe.

Integrates:
1. 75-dim state representation (25 individual stock/portfolio features + 25 cross-stock means + 25 cross-stock stds).
2. Deep Cross-Asset Multi-Head Attention Policy Network.
3. Turnover friction regularization and dynamic portfolio hold gating.
4. Trajectory-level policy gradient with leave-one-out baseline.
5. Multi-capital evaluation (₹500 to ₹200,000) and 2,000-resample block bootstrap.
"""

import argparse
import copy
import datetime
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

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from research.counterfactual_rl.market import Market, execute, project_scores, baseline

SEEDS = (42, 123, 456)
CAPITALS = [500, 5000, 50000, 100000, 200000]
HORIZON = 20
DEADBAND = 0.02


def extract_features_75(obs):
    """Extract 49 x 75 feature matrix: 25 asset features + 25 mean context + 25 std context."""
    stocks = obs['stocks']
    n = len(stocks)
    eq = max(float(obs['equity']), 1.0)
    cash = max(float(obs['cash']), 0.0)
    init_cap = max(float(obs['initial_capital']), 1.0)

    ranks = {
        k: np.argsort(np.argsort([r[k] for r in stocks], kind='stable'), kind='stable') / max(n - 1, 1)
        for k in ['return20', 'return60', 'return120', 'vol20']
    }

    x = []
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
        x.append(row)
    x = np.clip(np.array(x, dtype=np.float32), -10.0, 10.0)
    x_mean = np.broadcast_to(x.mean(axis=0, keepdims=True), x.shape)
    x_std = np.broadcast_to(x.std(axis=0, keepdims=True) + 1e-4, x.shape)
    feat = np.concatenate([x, x_mean, x_std], axis=1)
    return feat


class ConclusiveCrossAssetPolicy(nn.Module):
    """Multi-Head Attention State-Aware Policy over 49 assets."""
    def __init__(self, in_features=75, hidden=128, heads=4):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
        )

        layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=heads,
            dim_feedforward=hidden * 2,
            dropout=0.0,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=2)

        self.score_net = nn.Sequential(
            nn.Linear(hidden * 2, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, 1)
        )

    def forward(self, x):
        """x: (batch, 49, 75) -> scores: (batch, 49)"""
        h = self.encoder(x)
        z = self.transformer(h)

        # Cross-asset context
        z_mean = z.mean(dim=1, keepdim=True).expand(-1, 49, -1)
        combined = torch.cat([z, z_mean], dim=-1)

        raw_scores = self.score_net(combined).squeeze(-1)
        # Softplus for smooth non-negative allocation preference
        return F.softplus(raw_scores)

    def compute_weights(self, obs, deadband=DEADBAND, explore_noise=None):
        feat = extract_features_75(obs)
        tensor_x = torch.from_numpy(feat).unsqueeze(0).to(next(self.parameters()).device)

        with torch.no_grad():
            s = self(tensor_x)[0].cpu().numpy()

        if explore_noise is not None:
            s = np.maximum(0.0, s + explore_noise)

        # Focus capital on top 15 highest-conviction assets
        top15_idx = np.argsort(s)[-15:]
        filtered_scores = np.zeros(49)
        filtered_scores[top15_idx] = s[top15_idx]

        if filtered_scores.sum() < 1e-6:
            target_w = np.ones(49) / 49.0
        else:
            target_w = project_scores(filtered_scores)

        # Turnover control: If already holding positions, apply deadband
        curr_w = np.array([r['weight'] for r in obs['stocks']], dtype=np.float64)
        if curr_w.sum() > 0.1:
            diff = np.abs(target_w - curr_w)
            # If total desired turnover is below 12%, HOLD to avoid fees entirely
            if diff.sum() < 0.12:
                return None

            # Deadband on individual assets
            keep_mask = (diff < deadband) & (curr_w > 0.0)
            target_w[keep_mask] = curr_w[keep_mask]
            target_w = np.clip(target_w, 0.0, 0.10)
            if target_w.sum() > 1.0:
                target_w = project_scores(target_w)

        return target_w


def simulate_eval(m, policy, start, end, capital, deadband=DEADBAND):
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
            'weights': None if weights is None else weights.tolist(),
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


def train_seed(seed, m, device, total_epochs=70, batch_trajs=20):
    print(f"\n==========================================")
    print(f"Training Conclusive RL Policy - Seed {seed}")
    print(f"==========================================")

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    policy = ConclusiveCrossAssetPolicy().to(device)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=4e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs, eta_min=1e-5)

    first, last = m.bounds('2018-07-02', '2022-12-31')
    val_years = (2023, 2024)

    best_val_score = -float('inf')
    best_weights = None
    best_val_metrics = None
    history = []

    for epoch in range(1, total_epochs + 1):
        policy.train()
        explore_std = max(0.04, 0.25 * (1.0 - epoch / total_epochs))

        log_probs = []
        rewards = []

        for b in range(batch_trajs):
            t_start = int(np.random.randint(first, last - 100))
            cap = float(np.random.choice([50000, 100000, 200000]))
            steps = int(np.random.randint(3, 6))

            c = cap
            q = np.zeros(49, np.int64)
            p = np.zeros(2)
            t = t_start
            traj_values = [cap]
            traj_fees = 0.0

            step_lps = []

            for step in range(steps):
                obs = m.observe(t, c, q, cap, p)
                feat = extract_features_75(obs)
                tensor_x = torch.from_numpy(feat).unsqueeze(0).to(device)

                scores = policy(tensor_x)[0]
                noise = torch.randn_like(scores) * explore_std
                noisy_scores = F.relu(scores + noise)

                dist = torch.distributions.Normal(scores, explore_std)
                lp = dist.log_prob(noisy_scores).sum()
                step_lps.append(lp)

                s_np = noisy_scores.detach().cpu().numpy()
                top15 = np.argsort(s_np)[-15:]
                filt = np.zeros(49)
                filt[top15] = s_np[top15]

                if filt.sum() < 1e-6:
                    w = np.ones(49) / 49.0
                else:
                    w = project_scores(filt)

                curr_w = np.array([r['weight'] for r in obs['stocks']], dtype=np.float64)
                if curr_w.sum() > 0.1:
                    diff = np.abs(w - curr_w)
                    if diff.sum() < 0.12:
                        w = None
                    else:
                        keep_mask = (diff < DEADBAND) & (curr_w > 0.0)
                        w[keep_mask] = curr_w[keep_mask]
                        w = np.clip(w, 0.0, 0.10)
                        if w.sum() > 1.0:
                            w = project_scores(w)

                next_t = min(t + HORIZON, last)
                res = m.horizon(t, c, q, w, next_t, p)
                c, q, p = res['cash'], res['qty'], res['pending']
                traj_values.extend(res['values'].tolist())
                traj_fees += res['fees']
                t = next_t

            v = np.array(traj_values)
            tot_ret = float(v[-1] / v[0] - 1.0)
            dd = float((1.0 - v / np.maximum.accumulate(v)).max())
            fee_drag = traj_fees / v[0]

            # Net growth - drawdown penalty - 2x fee friction penalty
            reward = tot_ret - 0.4 * (dd ** 2) - 2.0 * fee_drag
            rewards.append(reward)
            log_probs.append(torch.stack(step_lps).mean())

        r_arr = np.array(rewards)
        baseline_r = (r_arr.sum() - r_arr) / max(len(r_arr) - 1, 1)
        adv = torch.from_numpy(r_arr - baseline_r).float().to(device)
        if adv.std() > 1e-6:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        loss = 0.0
        for lp, a in zip(log_probs, adv):
            loss -= lp * a
        loss = loss / batch_trajs

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if epoch % 5 == 0 or epoch == 1 or epoch == total_epochs:
            policy.eval()
            val_returns = {}
            for yr in val_years:
                res = simulate_eval(m, policy, f"{yr}-01-01", f"{yr}-12-31", 100000)
                val_returns[yr] = res['metrics']

            score = 0.5 * (np.log1p(val_returns[2023]['return']) + np.log1p(val_returns[2024]['return']))

            if score > best_val_score:
                best_val_score = score
                best_weights = copy.deepcopy(policy.state_dict())
                best_val_metrics = val_returns

            info = {
                'epoch': epoch,
                'loss': float(loss.item()),
                'val_score': float(score),
                'val_2023_ret': float(val_returns[2023]['return']),
                'val_2024_ret': float(val_returns[2024]['return']),
            }
            history.append(info)
            print(f"Epoch {epoch:2d}/{total_epochs} | Loss: {loss.item():6.4f} | "
                  f"Val 2023: {val_returns[2023]['return']*100:5.2f}% | "
                  f"Val 2024: {val_returns[2024]['return']*100:5.2f}% | "
                  f"Best Score: {best_val_score:6.4f}")

    policy.load_state_dict(best_weights)
    return policy, history, best_val_metrics


def run_test(seed, policy, m, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    policy.eval()

    start, end = "2025-01-01", "2026-01-30"

    from research.counterfactual_rl.market import evaluate as m_eval
    def eval_baseline(k, cap): return m_eval(m, baseline(k), start, end, cap)

    # ₹100,000 primary
    primary_sims = {}
    for b_name in ['buy_hold', 'equal_rebalanced', 'inverse_vol', 'momentum', 'cash']:
        primary_sims[b_name] = eval_baseline(b_name, 100000)

    rl_sim = simulate_eval(m, policy, start, end, 100000)
    primary_sims['state_aware_rl'] = rl_sim

    # Multi-capital
    cap_results = {}
    for cap in CAPITALS:
        r = simulate_eval(m, policy, start, end, cap)
        bh = eval_baseline('buy_hold', cap)
        eq = eval_baseline('equal_rebalanced', cap)
        cap_results[cap] = {
            'state_aware_rl': r['metrics'],
            'buy_hold': bh['metrics'],
            'equal_rebalanced': eq['metrics'],
        }
        print(f"Cap ₹{cap:7d} | RL: {r['metrics']['return']*100:6.2f}% | "
              f"Hold: {bh['metrics']['return']*100:6.2f}% | "
              f"Eq: {eq['metrics']['return']*100:6.2f}% | Fees: ₹{r['metrics']['fees']:.2f}")

    # Bootstrap
    rl_eq = primary_sims['state_aware_rl']['daily_equity']
    bootstraps = {}
    for comp in ['buy_hold', 'equal_rebalanced', 'momentum']:
        comp_eq = primary_sims[comp]['daily_equity']
        rng = np.random.default_rng(seed)
        a = np.r_[100000, rl_eq]
        b = np.r_[100000, comp_eq]
        diff = np.log(a[1:] / a[:-1]) - np.log(b[1:] / b[:-1])
        n = len(diff)
        b_samples = []
        for _ in range(2000):
            starts = rng.integers(0, n, int(np.ceil(n / 20)))
            ix = np.concatenate([(s + np.arange(20)) % n for s in starts])[:n]
            b_samples.append(float(diff[ix].sum()) * 100.0)
        ci = np.percentile(b_samples, [2.5, 97.5]).tolist()
        bootstraps[comp] = {
            'return_diff_pp': float((a[-1] / 100000 - b[-1] / 100000) * 100.0),
            'ci95': ci,
        }
        print(f"RL vs {comp:16s} Diff: {bootstraps[comp]['return_diff_pp']:+5.2f} pp | CI: [{ci[0]:+5.2f}, {ci[1]:+5.2f}] pp")

    res = {
        'seed': seed,
        'primary_metrics': {k: v['metrics'] for k, v in primary_sims.items()},
        'capital_stress': cap_results,
        'bootstraps': bootstraps,
        'rl_decisions': primary_sims['state_aware_rl']['decisions'],
        'rl_daily_equity': primary_sims['state_aware_rl']['daily_equity'],
    }
    (out_dir / f"seed{seed}_results.json").write_text(json.dumps(res, indent=2))
    return res


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', default='research/conclusive_final_runs')
    parser.add_argument('--epochs', type=int, default=60)
    args = parser.parse_args()

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    m = Market()
    master = {}

    for seed in SEEDS:
        p, hist, v_metrics = train_seed(seed, m, device, total_epochs=args.epochs, batch_trajs=20)
        torch.save(p.state_dict(), out_dir / f"policy_seed{seed}.pt")
        r = run_test(seed, p, m, out_dir)
        r['training_history'] = hist
        r['val_metrics'] = v_metrics
        master[seed] = r

    summary = {
        'timestamp': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'seeds': list(SEEDS),
        'results': {
            s: {
                'primary_metrics': master[s]['primary_metrics'],
                'capital_stress': master[s]['capital_stress'],
                'bootstraps': master[s]['bootstraps'],
            }
            for s in SEEDS
        }
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nCompleted! Saved to {out_dir / 'summary.json'}")


if __name__ == '__main__':
    main()
