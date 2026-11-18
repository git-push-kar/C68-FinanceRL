"""Evaluate trained InternVL3-2B + LoRA RL adapter on 2025-2026 Test Set.

Evaluates:
  1. Primary Test Evaluation at Rs 100,000 (2025-01-01 to 2026-01-30).
  2. Multi-capital sweep: Rs 5,000, Rs 50,000, Rs 100,000, Rs 200,000.
  3. Benchmarks comparison: Buy & Hold, Untrained InternVL3-2B, Equal Rebalanced.
  4. Circular block bootstrap (2,000 resamples) for 95% confidence intervals.
  5. Natural language rationale generation for investment decisions.
"""

import sys
import json
import time
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from peft import set_peft_model_state_dict
from safetensors.torch import load_file
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'repository'))

from research.counterfactual_rl.market import Market, baseline, evaluate as m_eval, project_scores
from research.train_internvl3_native_rl import InternVLPolicy, run_evaluation, LABELS, LABEL_VALUES

def bootstrap_difference(a, b, initial=100000.0, seed=42):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    min_len = min(len(a), len(b))
    a = np.r_[initial, a[:min_len]]
    b = np.r_[initial, b[:min_len]]
    diff = np.log(a[1:] / a[:-1]) - np.log(b[1:] / b[:-1])
    rng = np.random.default_rng(seed)
    n = len(diff)
    samples = []
    for _ in range(2000):
        starts = rng.integers(0, n, int(np.ceil(n / 20)))
        ix = np.concatenate([(s + np.arange(20)) % n for s in starts])[:n]
        samples.append(float(diff[ix].sum()))
    ci95 = np.quantile(samples, [0.025, 0.975]).tolist()
    return {
        'log_growth_diff': float(diff.sum()),
        'ci95': ci95,
        'p_superiority': float((np.array(samples) > 0).mean())
    }

def generate_decision_rationale(policy, obs, top_tickers):
    """Generate natural language rationale for the top picked stocks."""
    policy.lm.eval()
    stocks = {s['ticker']: s for s in obs['stocks']}
    rationales = []
    
    prompt = (
        "You are an expert quantitative portfolio manager. "
        "Explain concisely why each of the following top-selected Nifty stocks was allocated maximum capital "
        "based on their 20-day return, 60-day return, volatility, and price action:\n\n"
    )
    for ticker in top_tickers[:5]:
        s = stocks[ticker]
        prompt += (
            f"Stock {ticker.replace('.NS', '')}: Ret20={s['return20']*100:+.1f}%, "
            f"Ret60={s['return60']*100:+.1f}%, Vol20={s['vol20']*100:.1f}%, Price=Rs{s['historical_quote']:.0f}. "
            f"Investment rationale:"
        )
        input_ids = policy.tokenizer(prompt, return_tensors='pt')['input_ids'].to(policy.device)
        with torch.no_grad():
            gen = policy.lm.generate(
                input_ids=input_ids,
                max_new_tokens=40,
                temperature=0.2,
                top_p=0.9,
                do_sample=False,
                pad_token_id=policy.tokenizer.eos_token_id
            )
        output_text = policy.tokenizer.decode(gen[0][input_ids.shape[1]:], skip_special_tokens=True).strip()
        rationale = output_text.split('\n')[0].strip()
        rationales.append({'ticker': ticker, 'features': {
            'ret20': round(s['return20']*100, 2),
            'ret60': round(s['return60']*100, 2),
            'vol20': round(s['vol20']*100, 2),
            'price': round(s['historical_quote'], 2)
        }, 'rationale': rationale})
        prompt += f" {rationale}\n"
    return rationales

def main():
    m = Market()
    base_model_path = ROOT / 'PolyvalentFinance' / 'local_models' / 'InternVL3-2B'
    adapter_dir = ROOT / 'repository' / 'demo' / 'adapters' / 'internvl3_native_rl'
    weights_path = adapter_dir / 'finance_rl' / 'adapter_model.safetensors'

    print("=== EVALUATING TRAINED InternVL3-2B + LoRA RL POLICY ===")
    print(f"Base model: {base_model_path}")
    print(f"Adapter: {weights_path}")

    policy = InternVLPolicy(base_model_path)
    if weights_path.exists():
        print(f"Loading trained LoRA weights from {weights_path}...")
        sd = load_file(str(weights_path))
        set_peft_model_state_dict(policy.peft_model, sd, adapter_name='finance_rl')
        print("Trained LoRA weights successfully loaded into policy!")
    else:
        raise FileNotFoundError(f"Trained adapter weights not found at {weights_path}")

    test_start, test_end = '2025-01-01', '2026-01-30'
    capitals = [5000.0, 50000.0, 100000.0, 200000.0]

    # 1. Primary evaluation at Rs 100,000
    print("\n--- Evaluating on Test Set (2025-01-01 to 2026-01-30) at Rs 100,000 ---")
    trained_metrics = run_evaluation(m, policy, test_start, test_end, 100000.0)
    print(f"Trained RL Policy: Return = {trained_metrics['return']*100:+.2f}%, Sharpe = {trained_metrics['sharpe']:.3f}, MDD = {trained_metrics['max_drawdown']*100:.2f}%, Fees = Rs {trained_metrics['fees']:.2f}")

    # Buy & Hold
    bh_eval = m_eval(m, baseline('buy_hold'), test_start, test_end, 100000.0)
    bh_metrics = bh_eval['metrics']
    print(f"Buy & Hold:       Return = {bh_metrics['return']*100:+.2f}%, Sharpe = {bh_metrics['sharpe']:.3f}, MDD = {bh_metrics['max_drawdown']*100:.2f}%, Fees = Rs {bh_metrics['fees']:.2f}")

    # Equal Rebalanced
    eq_eval = m_eval(m, baseline('equal_rebalanced'), test_start, test_end, 100000.0)
    eq_metrics = eq_eval['metrics']
    print(f"Equal Rebalance:  Return = {eq_metrics['return']*100:+.2f}%, Sharpe = {eq_metrics['sharpe']:.3f}, MDD = {eq_metrics['max_drawdown']*100:.2f}%, Fees = Rs {eq_metrics['fees']:.2f}")

    # Untrained InternVL3-2B
    with policy.peft_model.disable_adapter():
        untrained_metrics = run_evaluation(m, policy, test_start, test_end, 100000.0)
    print(f"Untrained LLM:    Return = {untrained_metrics['return']*100:+.2f}%, Sharpe = {untrained_metrics['sharpe']:.3f}, MDD = {untrained_metrics['max_drawdown']*100:.2f}%, Fees = Rs {untrained_metrics['fees']:.2f}")

    # Bootstrap uncertainty
    boot_vs_bh = bootstrap_difference(trained_metrics['daily_equity'], bh_eval['daily_equity'])
    boot_vs_untrained = bootstrap_difference(trained_metrics['daily_equity'], untrained_metrics['daily_equity'])
    boot_vs_eq = bootstrap_difference(trained_metrics['daily_equity'], eq_eval['daily_equity'])

    print(f"\nBootstrap vs Buy & Hold: diff = {boot_vs_bh['log_growth_diff']:.4f}, 95% CI = {boot_vs_bh['ci95']}, p(superior) = {boot_vs_bh['p_superiority']*100:.1f}%")
    print(f"Bootstrap vs Untrained:  diff = {boot_vs_untrained['log_growth_diff']:.4f}, 95% CI = {boot_vs_untrained['ci95']}, p(superior) = {boot_vs_untrained['p_superiority']*100:.1f}%")

    # Multi-capital sweep
    print("\n--- Multi-Capital Scalability Sweep ---")
    cap_sweep = {}
    for cap in capitals:
        m_res = run_evaluation(m, policy, test_start, test_end, cap)
        bh_c = m_eval(m, baseline('buy_hold'), test_start, test_end, cap)['metrics']
        cap_sweep[str(int(cap))] = {
            'trained': {k: v for k, v in m_res.items() if k != 'daily_equity'},
            'buy_hold': bh_c
        }
        print(f"Capital Rs {cap:7,.0f} -> Trained: {m_res['return']*100:+.2f}% (Sharpe {m_res['sharpe']:.2f}, MDD {m_res['max_drawdown']*100:.1f}%, Fees Rs {m_res['fees']:.0f}) | B&H: {bh_c['return']*100:+.2f}% (Sharpe {bh_c['sharpe']:.2f}, MDD {bh_c['max_drawdown']*100:.1f}%)")

    # Generate sample decision rationale for first test rebalancing step
    print("\n--- Generating Sample Natural-Language Investment Rationales ---")
    t_test_st, _ = m.bounds(test_start, test_end)
    sample_obs = m.observe(t_test_st, 100000.0, np.zeros(49, np.int64), 100000.0)
    w = policy.compute_weights(sample_obs)
    top_indices = np.argsort(w)[-5:][::-1]
    top_tickers = [sample_obs['stocks'][i]['ticker'] for i in top_indices]
    rationales = generate_decision_rationale(policy, sample_obs, top_tickers)
    for r in rationales:
        print(f"[{r['ticker']}] Rationale: {r['rationale']}")

    # Save complete evaluation summary
    summary = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'test_period': f"{test_start} to {test_end}",
        'primary_capital_100k': {
            'trained_rl': {k: v for k, v in trained_metrics.items() if k != 'daily_equity'},
            'untrained_llm': {k: v for k, v in untrained_metrics.items() if k != 'daily_equity'},
            'buy_and_hold': bh_metrics,
            'equal_rebalanced': eq_metrics,
        },
        'bootstrap': {
            'vs_buy_and_hold': boot_vs_bh,
            'vs_untrained': boot_vs_untrained,
            'vs_equal_rebalanced': boot_vs_eq,
        },
        'capital_sweep': cap_sweep,
        'sample_rationales': rationales
    }

    out_file = adapter_dir / 'test_evaluation_summary.json'
    with open(out_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nEvaluation summary saved to {out_file}")

if __name__ == '__main__':
    main()
