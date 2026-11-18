"""Train InternVL3-2B + LoRA using Reinforcement Learning on 49-Stock Whole-Share Nifty Market.

Strict Chronological Splits:
  Train:      2018-07-02 to 2022-12-31 (1,113 trading sessions)
  Validation: 2023-01-01 to 2024-12-31 (492 trading sessions)
  Test:       2025-01-01 to 2026-01-30 (271 trading sessions)

Optimization:
  - Policy: Native InternVL3-2B Language Model with PEFT LoRA (r=16, alpha=32).
  - State: Numerical portfolio text prompt with all 49 stocks (Ret20, Ret60, Vol, Price, Afford).
  - Action: Categorical token distribution over 'A'..'G' for each stock.
  - Execution: Top-15 conviction, 10% position limit, 12% turnover hold-gating, 2% deadband.
  - Reward: Risk-adjusted excess return (alpha) minus volatility, drawdown, and fee drag.
"""

import os
import sys
import time
import copy
import json
import random
import hashlib
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoModel, AutoTokenizer
from torch.utils.checkpoint import checkpoint

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'repository'))

from research.counterfactual_rl.market import Market, baseline, evaluate as m_eval, project_scores

def checkpoint_layers(decoder):
    for layer in decoder.layers:
        original = layer.forward
        def wrap(forward):
            def call(*args, **kwargs):
                if torch.is_grad_enabled():
                    return checkpoint(forward, *args, use_reentrant=False, preserve_rng_state=True, **kwargs)
                return forward(*args, **kwargs)
            return call
        layer.forward = wrap(original)

LABELS = 'ABCDEFG'
LABEL_VALUES = np.array([0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0], dtype=np.float32)

class InternVLPolicy:
    def __init__(self, base_path, device='cuda'):
        self.device = device
        print(f"Loading tokenizer and base model from {base_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(base_path),
            trust_remote_code=True,
            local_files_only=True,
            use_fast=True
        )
        self.raw_model = AutoModel.from_pretrained(
            str(base_path),
            trust_remote_code=True,
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            use_flash_attn=False,
            local_files_only=True
        )
        
        lora_cfg = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.0,
            target_modules=r'(language_model\.model\.layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj|o_proj)|language_model\.lm_head)',
            bias='none'
        )
        self.peft_model = get_peft_model(self.raw_model, lora_cfg, adapter_name='finance_rl')
        base = self.peft_model.get_base_model()
        self.lm = base.language_model.to(device)
        checkpoint_layers(self.lm.model)

        self.label_tokens = [self.tokenizer.encode(x, add_special_tokens=False)[0] for x in LABELS]
        print(f"Policy initialized with PEFT LoRA. Trainable parameters: {sum(p.numel() for p in self.lm.parameters() if p.requires_grad):,}")

    def format_prompt(self, obs):
        stocks = sorted(obs['stocks'], key=lambda s: s['ticker'])
        lines = [
            "You are an expert quantitative portfolio manager managing all 49 Indian equities in the Nifty index.",
            f"Current Portfolio: Cash=Rs {obs['cash']:.0f}, Total Equity=Rs {obs['equity']:.0f}.",
            "Evaluate each stock's 20-day return, 60-day return, volatility, unit price, and cash affordability.",
            "Assign an investment grade from A (strongest sell) to G (strongest buy):"
        ]
        offsets = []
        text = "\n".join(lines)
        for s in stocks:
            ticker = s['ticker'].replace('.NS', '')
            q = s['historical_quote']
            r20 = s['return20'] * 100
            r60 = s['return60'] * 100
            vol = s['vol20'] * 100
            afford = s['affordable_shares_cash']
            line = f"\n{ticker}: Ret20={r20:+.1f}%, Ret60={r60:+.1f}%, Vol={vol:.1f}%, Price=Rs{q:.0f}, Afford={afford} sh. Grade:"
            text += line
            offsets.append(len(text))

        encoded = self.tokenizer(text, return_offsets_mapping=True, return_tensors='pt', add_special_tokens=False)
        ids = encoded['input_ids'].to(self.device)
        offset_mapping = encoded['offset_mapping'][0].tolist()

        target_indices = []
        for off in offsets:
            idx = next((i for i, (st, en) in enumerate(offset_mapping) if st < off <= en), len(offset_mapping) - 1)
            target_indices.append(idx)

        orig_tickers = [s['ticker'] for s in obs['stocks']]
        sorted_tickers = [s['ticker'] for s in stocks]
        reorder_map = [sorted_tickers.index(t) for t in orig_tickers]

        return ids, target_indices, reorder_map

    def forward_logits(self, obs):
        ids, target_indices, reorder_map = self.format_prompt(obs)
        emb = self.lm.get_input_embeddings()(ids)
        hidden = self.lm.model(inputs_embeds=emb, use_cache=False, return_dict=True).last_hidden_state
        logits = self.lm.lm_head(hidden[0, target_indices])
        grade_logits = logits[:, self.label_tokens].float()
        return grade_logits[reorder_map]

    def compute_weights(self, obs, deadband=0.02, hold_thresh=0.12, greedy=True):
        self.lm.eval()
        with torch.no_grad():
            grade_logits = self.forward_logits(obs)
            probs = F.softmax(grade_logits, dim=-1).cpu().numpy()
            if greedy:
                scores = (probs * LABEL_VALUES).sum(axis=-1)
            else:
                sampled_acts = np.array([np.random.choice(7, p=p) for p in probs])
                scores = LABEL_VALUES[sampled_acts]

        top15 = np.argsort(scores)[-15:]
        filt = np.zeros(49)
        filt[top15] = scores[top15]

        if filt.sum() < 1e-6:
            target_w = np.ones(49) / 49.0
        else:
            target_w = project_scores(filt)

        curr_w = np.array([r['weight'] for r in obs['stocks']], dtype=np.float64)
        if curr_w.sum() > 0.1:
            diff = np.abs(target_w - curr_w)
            if hold_thresh > 0 and diff.sum() < hold_thresh:
                return None
            if deadband > 0:
                keep = (diff < deadband) & (curr_w > 0.0)
                target_w[keep] = curr_w[keep]
                target_w = np.clip(target_w, 0.0, 0.10)
                if target_w.sum() > 1.0:
                    target_w = project_scores(target_w)
        return target_w


def run_evaluation(m, policy, start, end, capital=100000.0, deadband=0.02, hold_thresh=0.12):
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
        weights = policy.compute_weights(obs, deadband=deadband, hold_thresh=hold_thresh, greedy=True)
        next_t = min(t + 20, last)
        res = m.horizon(t, cash, qty, weights, next_t, pending)

        decisions.append({
            'date': str(m.dates[t]),
            'weights': None if weights is None else weights.tolist(),
            'equity': float(res['values'][-1]),
            'fees': float(res['fees']),
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

    return {
        'return': net_return,
        'end_equity': float(v[-1]),
        'sharpe': sharpe,
        'sortino': sortino,
        'max_drawdown': max_dd,
        'fees': total_fees,
        'slippage': total_slippage,
        'sessions': len(rets),
        'decisions_count': len(decisions),
        'daily_equity': daily_equity,
    }


def train_internvl_rl(args):
    m = Market()
    base_model_path = ROOT / 'PolyvalentFinance' / 'local_models' / 'InternVL3-2B'
    out_dir = ROOT / 'repository' / 'demo' / 'adapters' / 'internvl3_native_rl'
    out_dir.mkdir(parents=True, exist_ok=True)

    policy = InternVLPolicy(base_model_path)
    trainable_params = [p for p in policy.lm.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=1e-4)

    t_tr_st, t_tr_en = m.bounds('2018-07-02', '2022-12-31')
    starts = np.arange(t_tr_st, t_tr_en - 80, 10)
    capitals = [5000.0, 50000.0, 100000.0, 200000.0]

    print(f"--- Starting RL Training ---")
    print(f"Train window: {m.dates[t_tr_st]} to {m.dates[t_tr_en]} ({t_tr_en - t_tr_st + 1} sessions)")
    print(f"Candidate trajectory start indices: {len(starts)}")
    print(f"Epochs: {args.epochs}, Trajectory length: {args.traj_len} steps, LR: {args.lr}")

    # Evaluate untrained baseline on validation (2023-2024)
    print("\nEvaluating Untrained Policy on Validation (2023-2024)...")
    val_untrained = run_evaluation(m, policy, '2023-01-01', '2024-12-31', 100000.0)
    print(f"[Untrained Val] Return: {val_untrained['return']*100:.2f}%, Sharpe: {val_untrained['sharpe']:.3f}, MDD: {val_untrained['max_drawdown']*100:.2f}%, Fees: Rs {val_untrained['fees']:.2f}")

    best_val_score = val_untrained['return'] + 0.5 * val_untrained['sharpe'] - 0.2 * val_untrained['max_drawdown']
    best_metrics = val_untrained
    best_epoch = 0

    history = []
    t_start_training = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_t0 = time.time()
        policy.lm.train()
        optimizer.zero_grad()

        # Sample a trajectory
        t_start = int(np.random.choice(starts))
        capital = float(np.random.choice(capitals))
        cash = capital
        qty = np.zeros(49, np.int64)
        pending = np.zeros(2)

        traj_loss = 0.0
        traj_returns = []
        traj_alphas = []

        curr_t = t_start
        for step in range(args.traj_len):
            obs = m.observe(curr_t, cash, qty, capital, pending)
            grade_logits = policy.forward_logits(obs)
            probs = F.softmax(grade_logits, dim=-1)

            dist = torch.distributions.Categorical(probs=probs)
            actions = dist.sample() # (49,)
            log_probs = dist.log_prob(actions)

            sampled_scores = LABEL_VALUES[actions.detach().cpu().numpy()]
            top15 = np.argsort(sampled_scores)[-15:]
            filt = np.zeros(49)
            filt[top15] = sampled_scores[top15]
            target_w = project_scores(filt)

            # Hold-gating and deadband
            curr_w = np.array([r['weight'] for r in obs['stocks']], dtype=np.float64)
            if curr_w.sum() > 0.1:
                diff = np.abs(target_w - curr_w)
                if diff.sum() < 0.12:
                    target_w = None
                elif diff.max() < 0.02:
                    keep = (diff < 0.02) & (curr_w > 0.0)
                    target_w[keep] = curr_w[keep]
                    target_w = np.clip(target_w, 0.0, 0.10)
                    if target_w.sum() > 1.0:
                        target_w = project_scores(target_w)

            next_t = min(curr_t + 20, t_tr_en)
            res = m.horizon(curr_t, cash, qty, target_w, next_t, pending)

            # Stock return alpha
            p0 = m.close[curr_t]
            p1 = m.close[next_t]
            stock_rets = (p1 - p0) / p0
            mkt_ret = stock_rets.mean()
            stock_alpha = stock_rets - mkt_ret

            # Portfolio risk-adjusted reward
            v = res['values']
            step_ret = v[-1] / max(obs['equity'], 1e-6) - 1.0
            daily_rets = v[1:] / v[:-1] - 1.0
            daily_vol = daily_rets.std() if len(daily_rets) > 1 else 0.01
            fee_drag = (res['fees'] + res['slippage']) / max(obs['equity'], 1e-6)

            # Combined advantage
            centered_scores = sampled_scores - LABEL_VALUES.mean()
            stock_adv = centered_scores * stock_alpha
            port_adv = (step_ret - mkt_ret) - 0.20 * daily_vol * np.sqrt(252) - 2.0 * fee_drag
            raw_adv = stock_adv + 0.50 * port_adv
            norm_adv = (raw_adv - raw_adv.mean()) / (raw_adv.std() + 1e-4)
            adv_tensor = torch.tensor(norm_adv, dtype=torch.float32, device=policy.device)

            # Policy gradient + entropy
            step_pg_loss = -(log_probs * adv_tensor).mean()
            entropy = dist.entropy().mean()
            step_loss = step_pg_loss - 0.01 * entropy

            traj_loss = traj_loss + step_loss
            traj_returns.append(step_ret)
            traj_alphas.append(step_ret - mkt_ret)

            cash, qty, pending = res['cash'], res['qty'], res['pending']
            curr_t = next_t

        # Backward and step
        traj_loss = traj_loss / args.traj_len
        traj_loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
        optimizer.step()

        elapsed = time.time() - epoch_t0
        print(f"Epoch {epoch:2d}/{args.epochs} ({elapsed:.1f}s) | Loss: {traj_loss.item():.4f} | Ret: {np.mean(traj_returns)*100:+.2f}% | Alpha: {np.mean(traj_alphas)*100:+.2f}%")

        # Validation check every 5 epochs
        if epoch % 5 == 0 or epoch == args.epochs:
            print(f"--- Running Validation at Epoch {epoch} ---")
            val_metrics = run_evaluation(m, policy, '2023-01-01', '2024-12-31', 100000.0)
            val_score = val_metrics['return'] + 0.5 * val_metrics['sharpe'] - 0.2 * val_metrics['max_drawdown']
            print(f"[Val Epoch {epoch}] Return: {val_metrics['return']*100:.2f}%, Sharpe: {val_metrics['sharpe']:.3f}, MDD: {val_metrics['max_drawdown']*100:.2f}%, Score: {val_score:.4f}")

            history.append({
                'epoch': epoch,
                'loss': float(traj_loss.item()),
                'val_metrics': {k: v for k, v in val_metrics.items() if k != 'daily_equity'},
                'val_score': val_score
            })

            if val_score > best_val_score:
                best_val_score = val_score
                best_epoch = epoch
                best_metrics = val_metrics
                print(f"--> New best validation score: {val_score:.4f}! Saving checkpoint to {out_dir}...")
                policy.peft_model.save_pretrained(str(out_dir))
                with open(out_dir / 'best_val_metrics.json', 'w') as f:
                    json.dump({'best_epoch': epoch, 'val_score': val_score, **best_metrics}, f, indent=2)

    total_time = time.time() - t_start_training
    print(f"\nTraining completed in {total_time:.1f}s. Best validation epoch: {best_epoch} with score {best_val_score:.4f}.")

    with open(out_dir / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)

    return policy, out_dir


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--traj_len', type=int, default=3)
    parser.add_argument('--lr', type=float, default=5e-5)
    args = parser.parse_args()

    train_internvl_rl(args)
