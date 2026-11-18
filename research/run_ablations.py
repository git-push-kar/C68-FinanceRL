"""Ablation Studies for Conclusive State-Aware Full-Universe RL.

Ablations:
1. Execution Friction / Turnover Control:
   - Full Model: Deadband = 0.02, Hold-Gating Threshold = 0.12
   - No Hold-Gating: Deadband = 0.02, Hold-Gating = 0.0
   - No Deadband & No Gating: Deadband = 0.0, Hold-Gating = 0.0
   - Strict Gating: Deadband = 0.04, Hold-Gating = 0.20

2. State Representation Ablations (Trained from scratch):
   - Variant A (Full State-Aware): 75 dims (Technicals + Portfolio State + Cross-Asset Context)
   - Variant B (Technicals Only): 36 dims (20 Technicals + Cross-Asset Technical Stats), zero portfolio state
   - Variant C (No Cross-Asset Attention - Decoupled MLP): 75 dims input, but per-asset MLP without transformer layers

3. Capital Scaling Sweep:
   - Evaluated at ₹500, ₹5,000, ₹20,000, ₹50,000, ₹100,000, ₹200,000, ₹500,000.
"""

import os
import sys
import json
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

# Workspace imports
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'repository'))
sys.path.insert(0, str(ROOT / 'repository' / 'research'))

from research.counterfactual_rl.market import Market, execute, project_scores, baseline, evaluate as m_eval
from repository.research.train_conclusive_final import (
    extract_features_75,
    project_scores,
    ConclusiveCrossAssetPolicy,
    simulate_eval,
    HORIZON,
    DEADBAND
)


class DecoupledMLPPolicy(nn.Module):
    """Ablation architecture: Per-asset independent MLP without cross-asset self-attention."""
    def __init__(self, in_features=75, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, 1)
        )

    def forward(self, x):
        """x: (batch, 49, in_features) -> scores: (batch, 49)"""
        raw = self.net(x).squeeze(-1)
        return F.softplus(raw)

    def compute_weights(self, obs, deadband=DEADBAND, hold_thresh=0.12):
        feat = extract_features_75(obs)
        tensor_x = torch.from_numpy(feat).unsqueeze(0).to(next(self.parameters()).device)
        with torch.no_grad():
            s = self(tensor_x)[0].cpu().numpy()

        top15_idx = np.argsort(s)[-15:]
        filtered = np.zeros(49)
        filtered[top15_idx] = s[top15_idx]

        if filtered.sum() < 1e-6:
            target_w = np.ones(49) / 49.0
        else:
            target_w = project_scores(filtered)

        curr_w = np.array([r['weight'] for r in obs['stocks']], dtype=np.float64)
        if curr_w.sum() > 0.1:
            diff = np.abs(target_w - curr_w)
            if hold_thresh > 0 and diff.sum() < hold_thresh:
                return None
            if deadband > 0:
                keep_mask = (diff < deadband) & (curr_w > 0.0)
                target_w[keep_mask] = curr_w[keep_mask]
                target_w = np.clip(target_w, 0.0, 0.10)
                if target_w.sum() > 1.0:
                    target_w = project_scores(target_w)
        return target_w


def extract_features_technicals_only(obs):
    """Extract features with portfolio-level state zeroed out."""
    feat = extract_features_75(obs)
    # Feature indices:
    # 0: price, 1: shares, 2: pos_val, 3: weight, 4: afford_val,
    # 5-11: returns, vol, dd, vol_ratio
    # 12-19: capital, cash, eq, cash_ratio, max_w, fees, slip, has_pos
    # 20-23: ranks
    # 24: price_ratio
    # Zero out portfolio-dependent features across x, x_mean, x_std
    # Portfolio indices in 25-dim slice: 1, 2, 3, 4, 12, 13, 14, 15, 19, 24
    port_indices = [1, 2, 3, 4, 12, 13, 14, 15, 19, 24]
    all_port_indices = []
    for offset in [0, 25, 50]:
        all_port_indices.extend([i + offset for i in port_indices])
    
    feat[:, all_port_indices] = 0.0
    return feat


class PolicyWrapper:
    def __init__(self, policy, deadband=0.02, hold_thresh=0.12, feat_fn=extract_features_75):
        self.policy = policy
        self.deadband = deadband
        self.hold_thresh = hold_thresh
        self.feat_fn = feat_fn

    def compute_weights(self, obs, deadband=None):
        db = self.deadband if deadband is None else deadband
        feat = self.feat_fn(obs)
        tensor_x = torch.from_numpy(feat).unsqueeze(0).to(next(self.policy.parameters()).device)
        with torch.no_grad():
            s = self.policy(tensor_x)[0].cpu().numpy()

        top15 = np.argsort(s)[-15:]
        filt = np.zeros(49)
        filt[top15] = s[top15]

        if filt.sum() < 1e-6:
            target_w = np.ones(49) / 49.0
        else:
            target_w = project_scores(filt)

        curr_w = np.array([r['weight'] for r in obs['stocks']], dtype=np.float64)
        if curr_w.sum() > 0.1:
            diff = np.abs(target_w - curr_w)
            if self.hold_thresh > 0 and diff.sum() < self.hold_thresh:
                return None
            if db > 0:
                keep = (diff < db) & (curr_w > 0.0)
                target_w[keep] = curr_w[keep]
                target_w = np.clip(target_w, 0.0, 0.10)
                if target_w.sum() > 1.0:
                    target_w = project_scores(target_w)
        return target_w


def train_policy_custom(policy_cls, feat_fn, m, device, seed=42, total_epochs=50, batch_trajs=15):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    policy = policy_cls().to(device)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=4e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs, eta_min=1e-5)

    first, last = m.bounds('2018-07-02', '2022-12-31')

    best_val_score = -float('inf')
    best_state = None

    for epoch in range(1, total_epochs + 1):
        policy.train()
        explore_std = max(0.04, 0.25 * (1.0 - epoch / total_epochs))

        log_probs = []
        rewards = []

        for _ in range(batch_trajs):
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

            for _ in range(steps):
                obs = m.observe(t, c, q, cap, p)
                feat = feat_fn(obs)
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
                w = np.ones(49) / 49.0 if filt.sum() < 1e-6 else project_scores(filt)

                curr_w = np.array([r['weight'] for r in obs['stocks']], dtype=np.float64)
                if curr_w.sum() > 0.1:
                    diff = np.abs(w - curr_w)
                    if diff.sum() < 0.12:
                        w = None

                next_t = min(t + HORIZON, last)
                res = m.horizon(t, c, q, w, next_t, p)
                c, q, p = res['cash'], res['qty'], res['pending']
                t = next_t
                traj_values.append(float(res['values'][-1]))
                traj_fees += res['fees']

            vals = np.array(traj_values)
            step_rets = vals[1:] / vals[:-1] - 1.0
            mean_r = step_rets.mean()
            std_r = max(step_rets.std(), 1e-6)
            sharpe_proxy = mean_r / std_r
            fee_penalty = (traj_fees / cap) * 2.0
            total_r = (vals[-1] / cap - 1.0)
            reward = total_r + 0.1 * sharpe_proxy - fee_penalty

            log_probs.append(torch.stack(step_lps).sum())
            rewards.append(reward)

        log_probs_t = torch.stack(log_probs)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=device)
        norm_rewards = (rewards_t - rewards_t.mean()) / (rewards_t.std() + 1e-6)

        loss = -(log_probs_t * norm_rewards.detach()).mean()
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        # Validation check every 5 epochs
        if epoch % 5 == 0:
            policy.eval()
            wrapper = PolicyWrapper(policy, deadband=0.02, hold_thresh=0.12, feat_fn=feat_fn)
            val_res = simulate_eval(m, wrapper, '2023-01-02', '2024-12-31', 100000.0)
            v_score = val_res['metrics']['return'] + 0.5 * val_res['metrics']['sharpe']
            if v_score > best_val_score:
                best_val_score = v_score
                best_state = {k: v.cpu().clone() for k, v in policy.state_dict().items()}

    policy.load_state_dict(best_state)
    policy.eval()
    return policy


def main():
    print("=" * 60)
    print("Running Rigorous Ablations for Journal Research Paper")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    m = Market()
    print("Loaded Market across all 49 assets.")

    saved_dir = ROOT / 'repository' / 'research' / 'conclusive_final_runs'
    checkpoint_path = saved_dir / 'policy_seed42.pt'
    
    # Load primary trained model (Seed 42)
    primary_policy = ConclusiveCrossAssetPolicy().to(device)
    primary_policy.load_state_dict(torch.load(checkpoint_path, map_location=device))
    primary_policy.eval()
    print("Loaded primary Seed 42 checkpoint.")

    test_start, test_end = '2025-01-01', '2026-01-30'
    baseline_bnh = m_eval(m, baseline('buy_hold'), test_start, test_end, 100000.0)['metrics']
    baseline_eq = m_eval(m, baseline('equal_rebalanced'), test_start, test_end, 100000.0)['metrics']

    ablations = {}

    # -------------------------------------------------------------
    # 1. Execution Friction & Turnover Control Ablations
    # -------------------------------------------------------------
    print("\n--- Running Execution Friction & Turnover Control Ablations ---")
    turnover_configs = {
        "full_control": {"deadband": 0.02, "hold_thresh": 0.12, "desc": "Full Framework (Deadband 2%, Hold-Gate 12%)"},
        "no_hold_gate": {"deadband": 0.02, "hold_thresh": 0.0, "desc": "No Hold-Gate (Rebalance every horizon, Deadband 2%)"},
        "zero_control": {"deadband": 0.0, "hold_thresh": 0.0, "desc": "Zero Control (Blind rebalancing, 0% Deadband, 0% Gate)"},
        "strict_gate":  {"deadband": 0.04, "hold_thresh": 0.20, "desc": "Strict Hold-Gate (Deadband 4%, Hold-Gate 20%)"},
    }

    turnover_results = {}
    for name, cfg in turnover_configs.items():
        wrapper = PolicyWrapper(primary_policy, deadband=cfg["deadband"], hold_thresh=cfg["hold_thresh"])
        res = simulate_eval(m, wrapper, test_start, test_end, 100000.0)
        met = res['metrics']
        turnover_results[name] = {
            "description": cfg["desc"],
            "return": met['return'],
            "end_equity": met['end_equity'],
            "sharpe": met['sharpe'],
            "sortino": met['sortino'],
            "max_drawdown": met['max_drawdown'],
            "fees": met['fees'],
            "slippage": met['slippage'],
            "trades_count": met['decisions_count'],
        }
        print(f"[{name}] Return: {met['return']*100:.2f}%, Sharpe: {met['sharpe']:.3f}, MDD: {met['max_drawdown']*100:.2f}%, Fees: Rs {met['fees']:.2f}")

    ablations['turnover_control'] = turnover_results

    # -------------------------------------------------------------
    # 2. State Representation & Architecture Ablations
    # -------------------------------------------------------------
    print("\n--- Running Architecture & State Representation Ablations ---")
    
    # 2.1 State Representation: Technicals Only (Zero Portfolio State)
    print("\nTraining Ablation Variant 1: Technicals Only (No Portfolio State)...")
    tech_policy = train_policy_custom(
        ConclusiveCrossAssetPolicy,
        extract_features_technicals_only,
        m, device, seed=42, total_epochs=50
    )
    wrapper_tech = PolicyWrapper(tech_policy, feat_fn=extract_features_technicals_only)
    res_tech = simulate_eval(m, wrapper_tech, test_start, test_end, 100000.0)
    met_tech = res_tech['metrics']
    print(f"[Technicals Only] Return: {met_tech['return']*100:.2f}%, Sharpe: {met_tech['sharpe']:.3f}, MDD: {met_tech['max_drawdown']*100:.2f}%, Fees: Rs {met_tech['fees']:.2f}")

    # 2.2 Architecture: Decoupled MLP (No Cross-Asset Self-Attention)
    print("\nTraining Ablation Variant 2: Decoupled Per-Asset MLP (No Self-Attention)...")
    mlp_policy = train_policy_custom(
        DecoupledMLPPolicy,
        extract_features_75,
        m, device, seed=42, total_epochs=50
    )
    wrapper_mlp = PolicyWrapper(mlp_policy, feat_fn=extract_features_75)
    res_mlp = simulate_eval(m, wrapper_mlp, test_start, test_end, 100000.0)
    met_mlp = res_mlp['metrics']
    print(f"[Decoupled MLP] Return: {met_mlp['return']*100:.2f}%, Sharpe: {met_mlp['sharpe']:.3f}, MDD: {met_mlp['max_drawdown']*100:.2f}%, Fees: Rs {met_mlp['fees']:.2f}")

    ablations['architecture_and_state'] = {
        "full_state_attention": turnover_results["full_control"],
        "technicals_only_no_portfolio": {
            "description": "Technicals Only (Portfolio Cash, Holdings & Affordability Zeroed)",
            "return": met_tech['return'],
            "end_equity": met_tech['end_equity'],
            "sharpe": met_tech['sharpe'],
            "sortino": met_tech['sortino'],
            "max_drawdown": met_tech['max_drawdown'],
            "fees": met_tech['fees'],
            "slippage": met_tech['slippage'],
        },
        "decoupled_mlp_no_attention": {
            "description": "Decoupled Per-Asset MLP (No Multi-Head Self-Attention)",
            "return": met_mlp['return'],
            "end_equity": met_mlp['end_equity'],
            "sharpe": met_mlp['sharpe'],
            "sortino": met_mlp['sortino'],
            "max_drawdown": met_mlp['max_drawdown'],
            "fees": met_mlp['fees'],
            "slippage": met_mlp['slippage'],
        }
    }

    # -------------------------------------------------------------
    # 3. Capital Scale Granularity Sweep
    # -------------------------------------------------------------
    print("\n--- Running Capital Scale Granularity Sweep ---")
    capital_tiers = [500, 2000, 5000, 10000, 20000, 50000, 100000, 200000, 500000, 1000000]
    cap_sweep = {}
    wrapper_full = PolicyWrapper(primary_policy)

    for cap in capital_tiers:
        r_rl = simulate_eval(m, wrapper_full, test_start, test_end, float(cap))['metrics']
        r_bnh = m_eval(m, baseline('buy_hold'), test_start, test_end, float(cap))['metrics']
        r_eq = m_eval(m, baseline('equal_rebalanced'), test_start, test_end, float(cap))['metrics']

        cap_sweep[str(cap)] = {
            "rl_return": r_rl['return'],
            "rl_equity": r_rl['end_equity'],
            "rl_sharpe": r_rl['sharpe'],
            "rl_mdd": r_rl['max_drawdown'],
            "rl_fees": r_rl['fees'],
            "bnh_return": r_bnh['return'],
            "bnh_equity": r_bnh['end_equity'],
            "eq_return": r_eq['return'],
            "eq_equity": r_eq['end_equity'],
        }
        print(f"Capital Rs {cap:8,d} -> RL Return: {r_rl['return']*100:6.2f}% | B&H: {r_bnh['return']*100:6.2f}% | RL Fees: Rs {r_rl['fees']:8.2f}")

    ablations['capital_sweep'] = cap_sweep

    # -------------------------------------------------------------
    # 4. Asset Selection & Top Holding Analysis
    # -------------------------------------------------------------
    print("\n--- Analyzing Top Selected Assets across 2025 ---")
    res_full = simulate_eval(m, wrapper_full, test_start, test_end, 100000.0)
    decisions = res_full['decisions']
    
    # Aggregate weights allocated to each stock over the year
    stock_names = m.tickers
    weight_matrix = []
    for d in decisions:
        if d['weights'] is not None:
            weight_matrix.append(d['weights'])
    
    avg_weights = np.mean(weight_matrix, axis=0) if len(weight_matrix) > 0 else np.zeros(49)
    top_stocks_idx = np.argsort(avg_weights)[::-1]

    # Calculate 2025 raw return for each stock
    t0, t1 = m.bounds(test_start, test_end)
    stock_returns_2025 = (m.close[t1] / m.close[t0] - 1.0)

    asset_analysis = []
    for idx in top_stocks_idx[:15]:
        asset_analysis.append({
            "ticker": stock_names[idx],
            "average_allocated_weight": float(avg_weights[idx]),
            "raw_2025_return": float(stock_returns_2025[idx]),
            "initial_price": float(m.close[t0, idx]),
            "final_price": float(m.close[t1, idx]),
        })

    ablations['top_asset_selection'] = asset_analysis

    # Save to JSON
    out_file = saved_dir / 'ablations_summary.json'
    with open(out_file, 'w') as f:
        json.dump(ablations, f, indent=2)
    print(f"\nAblation results successfully saved to: {out_file}")


if __name__ == '__main__':
    main()
