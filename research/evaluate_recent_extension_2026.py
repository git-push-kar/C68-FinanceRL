"""Evaluation of Trained InternVL3-2B + LoRA RL, Untrained LLM, and Buy & Hold
on Recent Market Data (February 2, 2026 to November 17, 2026, 195 Trading Sessions).

Evaluates across multiple initial capital tiers:
  Rs 5,000, Rs 10,000, Rs 25,000, Rs 50,000, Rs 75,000, Rs 100,000, Rs 150,000, Rs 200,000, Rs 500,000.
"""

import sys
import json
import time
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from pathlib import Path
from peft import set_peft_model_state_dict
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'repository'))

from research.counterfactual_rl.market import Market, baseline, evaluate as m_eval, project_scores, EVENTS
from research.train_internvl3_native_rl import InternVLPolicy, run_evaluation, LABELS, LABEL_VALUES


class ExtendedMarket(Market):
    """Continuous Nifty 49 market environment extending through November 17, 2026."""
    def __init__(self):
        super().__init__()
        FULL_ROOT = ROOT / 'research' / 'full_universe_rl'
        exec_dir = FULL_ROOT / 'execution_fields'
        op = np.load(FULL_ROOT / 'original_panel.npz', allow_pickle=True)
        hist_dates = list(op['dates'])
        
        f_ref = pd.read_csv(exec_dir / 'RELIANCE.NS.csv', index_col=0)
        f_ref.index = pd.to_datetime(f_ref.index, utc=True).tz_convert('Asia/Kolkata').tz_localize(None).normalize()
        ext_sub = f_ref.loc['2026-02-01':'2026-11-17']
        ext_dates = [np.datetime64(d.strftime('%Y-%m-%d')) for d in ext_sub.index]
        
        self.dates = np.array(hist_dates + ext_dates)
        n_dates = len(self.dates)
        n_tickers = len(self.tickers)
        
        open_p = np.full((n_dates, n_tickers), np.nan)
        close_p = np.full((n_dates, n_tickers), np.nan)
        adj_close_p = np.full((n_dates, n_tickers), np.nan)
        vol_p = np.zeros((n_dates, n_tickers))
        div_p = np.zeros((n_dates, n_tickers))
        split_p = np.ones((n_dates, n_tickers))
        
        n_hist = len(hist_dates)
        adj_close_p[:n_hist] = op['adjusted_close']
        vol_p[:n_hist] = op['volume']
        
        for i, ticker in enumerate(self.tickers):
            f = pd.read_csv(exec_dir / f'{ticker}.csv', index_col=0)
            f.index = pd.to_datetime(f.index, utc=True).tz_convert('Asia/Kolkata').tz_localize(None).normalize()
            split = f['Stock Splits'].replace(0, 1).to_numpy(float)
            future = np.cumprod(split[::-1])[::-1] / split
            op_arr = f.Open.to_numpy() * future
            cl_arr = f.Close.to_numpy() * future
            div_arr = f.Dividends.to_numpy() * future
            
            for ev in EVENTS:
                if ev['parent'] == ticker:
                    j = f.index.get_loc(pd.Timestamp(ev['prior']))
                    factor = ev['prior_close'] / cl_arr[j]
                    before = f.index < pd.Timestamp(ev['ex'])
                    op_arr[before] *= factor
                    cl_arr[before] *= factor
                    
            lookup = f.index.get_indexer(pd.to_datetime(self.dates))
            valid = lookup >= 0
            ix = lookup[valid]
            open_p[valid, i] = op_arr[ix]
            close_p[valid, i] = cl_arr[ix]
            div_p[valid, i] = div_arr[ix]
            split_p[valid, i] = split[ix]
            
            ext_lookup = f.index.get_indexer(pd.to_datetime(ext_dates))
            ext_valid = ext_lookup >= 0
            ext_ix = ext_lookup[ext_valid]
            adj_close_p[n_hist:n_hist+len(ext_dates), i] = f['Adj Close'].to_numpy()[ext_ix]
            vol_p[n_hist:n_hist+len(ext_dates), i] = f['Volume'].to_numpy()[ext_ix]
            
        self.open = open_p
        self.close = close_p
        self.prices = adj_close_p
        self.volume = vol_p
        self.dividends = div_p
        self.splits = split_p


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


def main():
    print("=== INITIALIZING EXTENDED MARKET ENVIRONMENT (Feb 2 - Nov 17, 2026) ===")
    m = ExtendedMarket()
    test_start = '2026-02-02'
    test_end = '2026-11-17'
    
    first, last = m.bounds(test_start, test_end)
    print(f"Evaluation window: {m.dates[first]} to {m.dates[last]} ({last - first} trading sessions)")
    
    base_model_path = ROOT.parent / 'PolyvalentFinance' / 'local_models' / 'InternVL3-2B'
    if not base_model_path.exists():
        base_model_path = ROOT / 'models' / 'InternVL3-2B'
        
    adapter_path = ROOT / 'demo' / 'adapters' / 'internvl3_state_rl' / 'adapter_model.safetensors'
    if not adapter_path.exists():
        adapter_path = ROOT / 'demo' / 'adapters' / 'internvl3_native_rl' / 'finance_rl' / 'adapter_model.safetensors'
        
    print(f"Base model: {base_model_path}")
    print(f"Adapter:    {adapter_path}")
    
    # Capital levels to sweep
    capitals = [5000.0, 10000.0, 25000.0, 50000.0, 75000.0, 100000.0, 150000.0, 200000.0, 500000.0]
    
    print("\n--- Initializing InternVL Policy ---")
    policy = InternVLPolicy(base_model_path)
    
    # 1. Evaluate Untrained Policy Across Capitals
    print("\n" + "="*70)
    print("1. EVALUATING UNTRAINED InternVL3-2B BASELINE")
    print("="*70)
    untrained_results = {}
    for cap in capitals:
        t0 = time.time()
        res = run_evaluation(m, policy, test_start, test_end, capital=cap)
        dur = time.time() - t0
        untrained_results[str(int(cap))] = res
        print(f"Untrained [Rs {int(cap):>7,d}]: Return={res['return']*100:>+6.2f}%, Sharpe={res['sharpe']:>5.3f}, MDD={res['max_drawdown']*100:>5.2f}%, Fees=Rs {res['fees']:>7.2f} ({dur:.1f}s)")
        
    # 2. Load Trained LoRA Weights and Evaluate
    print("\n" + "="*70)
    print("2. LOADING TRAINED LoRA WEIGHTS AND EVALUATING TRAINED RL POLICY")
    print("="*70)
    sd = load_file(str(adapter_path))
    set_peft_model_state_dict(policy.peft_model, sd, adapter_name='finance_rl')
    print("Trained LoRA weights successfully loaded!")
    
    trained_results = {}
    for cap in capitals:
        t0 = time.time()
        res = run_evaluation(m, policy, test_start, test_end, capital=cap)
        dur = time.time() - t0
        trained_results[str(int(cap))] = res
        print(f"Trained RL [Rs {int(cap):>7,d}]: Return={res['return']*100:>+6.2f}%, Sharpe={res['sharpe']:>5.3f}, MDD={res['max_drawdown']*100:>5.2f}%, Fees=Rs {res['fees']:>7.2f} ({dur:.1f}s)")
        
    # 3. Evaluate Buy & Hold Strategy Across Capitals
    print("\n" + "="*70)
    print("3. EVALUATING PASSIVE BUY & HOLD BENCHMARK")
    print("="*70)
    bh_results = {}
    for cap in capitals:
        res_bh = m_eval(m, baseline('buy_hold'), test_start, test_end, capital=cap)
        bh_metrics = res_bh['metrics']
        bh_metrics['daily_equity'] = res_bh['daily_equity']
        bh_results[str(int(cap))] = bh_metrics
        print(f"Buy & Hold [Rs {int(cap):>7,d}]: Return={bh_metrics['return']*100:>+6.2f}%, Sharpe={bh_metrics['sharpe']:>5.3f}, MDD={bh_metrics['max_drawdown']*100:>5.2f}%, Fees=Rs {bh_metrics['fees']:>7.2f}")
        
    # 4. Evaluate Equal-Weight Rebalancing Strategy Across Capitals
    print("\n" + "="*70)
    print("4. EVALUATING EQUAL-WEIGHT REBALANCED BENCHMARK")
    print("="*70)
    eq_results = {}
    for cap in capitals:
        res_eq = m_eval(m, baseline('equal_rebalanced'), test_start, test_end, capital=cap)
        eq_metrics = res_eq['metrics']
        eq_metrics['daily_equity'] = res_eq['daily_equity']
        eq_results[str(int(cap))] = eq_metrics
        print(f"Equal Rebal [Rs {int(cap):>7,d}]: Return={eq_metrics['return']*100:>+6.2f}%, Sharpe={eq_metrics['sharpe']:>5.3f}, MDD={eq_metrics['max_drawdown']*100:>5.2f}%, Fees=Rs {eq_metrics['fees']:>7.2f}")

    # 5. Bootstrap Comparison at Primary Rs 100,000 Capital
    print("\n" + "="*70)
    print("5. BOOTSTRAP STATISTICAL INFERENCE (Primary Rs 100,000 Capital)")
    print("="*70)
    boot_bh = bootstrap_difference(trained_results['100000']['daily_equity'], bh_results['100000']['daily_equity'], initial=100000.0)
    boot_unt = bootstrap_difference(trained_results['100000']['daily_equity'], untrained_results['100000']['daily_equity'], initial=100000.0)
    boot_eq = bootstrap_difference(trained_results['100000']['daily_equity'], eq_results['100000']['daily_equity'], initial=100000.0)

    print(f"Trained RL vs Buy & Hold:      Delta={boot_bh['log_growth_diff']*100:+.2f}%, 95% CI=[{boot_bh['ci95'][0]*100:+.2f}%, {boot_bh['ci95'][1]*100:+.2f}%], P(Trained > BH)={boot_bh['p_superiority']*100:.1f}%")
    print(f"Trained RL vs Untrained LLM:   Delta={boot_unt['log_growth_diff']*100:+.2f}%, 95% CI=[{boot_unt['ci95'][0]*100:+.2f}%, {boot_unt['ci95'][1]*100:+.2f}%], P(Trained > Untrained)={boot_unt['p_superiority']*100:.1f}%")
    print(f"Trained RL vs Equal-Rebalance: Delta={boot_eq['log_growth_diff']*100:+.2f}%, 95% CI=[{boot_eq['ci95'][0]*100:+.2f}%, {boot_eq['ci95'][1]*100:+.2f}%], P(Trained > EQ)={boot_eq['p_superiority']*100:.1f}%")

    # Generate Decision Rationales for Top Allocations on Recent Dates
    print("\n" + "="*70)
    print("6. GENERATING VERIFIABLE DECISION RATIONALES ON RECENT MARKET DATA")
    print("="*70)
    policy.lm.eval()
    obs = m.observe(last - 1, 100000.0, np.zeros(49, np.int64), 100000.0, np.zeros(2))
    logits = policy.forward_logits(obs)
    probs = F.softmax(logits, dim=-1).detach().cpu().numpy()
    scores = (probs * LABEL_VALUES).sum(axis=-1)
    top_indices = np.argsort(scores)[-5:][::-1]
    
    rationales = []
    for idx in top_indices:
        ticker = m.tickers[idx]
        s = obs['stocks'][idx]
        t_clean = ticker.replace('.NS', '')
        prompt = (
            f"Stock {t_clean}: Ret20={s['return20']*100:+.1f}%, Ret60={s['return60']*100:+.1f}%, "
            f"Vol={s['vol20']*100:.1f}%, Price=Rs{s['historical_quote']:.0f}. "
            f"Explain concisely in one sentence why this stock was allocated capital based on its momentum and risk:"
        )
        r_ids = policy.tokenizer(prompt, return_tensors='pt')['input_ids'].to(policy.device)
        with torch.no_grad():
            gen = policy.lm.generate(input_ids=r_ids, max_new_tokens=35, do_sample=False, pad_token_id=policy.tokenizer.eos_token_id)
        gen_text = policy.tokenizer.decode(gen[0][r_ids.shape[1]:], skip_special_tokens=True).strip().split('\n')[0].strip()
        rationales.append({'ticker': ticker, 'features': {
            'ret20': round(s['return20']*100, 2),
            'ret60': round(s['return60']*100, 2),
            'vol': round(s['vol20']*100, 2),
            'price': round(s['historical_quote'], 2)
        }, 'rationale': gen_text})
        print(f"[{ticker}] {gen_text}")

    # Prepare complete payload
    summary = {
        'evaluation_period': {
            'start_date': str(m.dates[first]),
            'end_date': str(m.dates[last]),
            'trading_sessions': int(last - first),
            'market_universe': 'Nifty 50 (all 49 continuous constituents)'
        },
        'primary_capital_100k': {
            'trained_rl': {k: v for k, v in trained_results['100000'].items() if k != 'daily_equity'},
            'untrained_llm': {k: v for k, v in untrained_results['100000'].items() if k != 'daily_equity'},
            'buy_and_hold': {k: v for k, v in bh_results['100000'].items() if k != 'daily_equity'},
            'equal_rebalanced': {k: v for k, v in eq_results['100000'].items() if k != 'daily_equity'},
            'bootstrap_inference': {
                'vs_buy_and_hold': boot_bh,
                'vs_untrained_llm': boot_unt,
                'vs_equal_rebalance': boot_eq
            }
        },
        'multi_capital_sweep': {
            'capitals': capitals,
            'trained_rl': {cap: {k: v for k, v in res.items() if k != 'daily_equity'} for cap, res in trained_results.items()},
            'untrained_llm': {cap: {k: v for k, v in res.items() if k != 'daily_equity'} for cap, res in untrained_results.items()},
            'buy_and_hold': {cap: {k: v for k, v in res.items() if k != 'daily_equity'} for cap, res in bh_results.items()},
            'equal_rebalanced': {cap: {k: v for k, v in res.items() if k != 'daily_equity'} for cap, res in eq_results.items()},
        },
        'rationales_sample': rationales
    }
    
    out_file1 = ROOT / 'research' / 'recent_2026_evaluation_summary.json'
    out_file2 = ROOT / 'demo' / 'adapters' / 'internvl3_native_rl' / 'recent_2026_evaluation_summary.json'
    out_file1.write_text(json.dumps(summary, indent=2))
    out_file2.parent.mkdir(parents=True, exist_ok=True)
    out_file2.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote results summary to {out_file1} and {out_file2}")


if __name__ == '__main__':
    main()
