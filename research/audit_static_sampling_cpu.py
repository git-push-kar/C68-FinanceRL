"""Sampled-action sensitivity of a verified constant native policy, using saved probabilities."""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

p = argparse.ArgumentParser()
p.add_argument('--run', required=True)
p.add_argument('--source', required=True)
p.add_argument('--data', required=True)
p.add_argument('--start', required=True)
p.add_argument('--end', required=True)
a = p.parse_args()
root = Path(__file__).parent
run = Path(a.run).resolve()
source = Path(a.source).resolve()
data = Path(a.data).resolve()
out = run / 'static_sampled_sensitivity.json'
if out.exists():
    raise FileExistsError(out)
if a.end > '2024-12-31':
    raise ValueError('Development-only diagnostic')
sys.path.insert(0, str(source))
os.chdir(source)
from candidate_market import Market, evaluate
from metrics import metrics_from_infos
from execution_policy import wait_for_compute_window

def read(path):
    return json.loads(path.read_text())

assert read(run / 'status.json')['status'] == 'completed'
assert read(run / 'reload_audit.json')['status'] == 'passed'
step = read(run / 'selection.json')['selected']['step']
market = Market(str(data), a.start, a.end)
probabilities = {}
parity = {}
for name, checkpoint in [('initial', 0), ('trained', step)]:
    expected = read(run / f'development_{checkpoint}.json')
    matrix = np.asarray(expected['probabilities'], dtype=float)
    if np.max(abs(matrix - matrix[0])) > 1e-12:
        raise ValueError('Policy is not constant; cannot substitute saved probabilities')
    probs = matrix[0] / matrix[0].sum()
    actual = evaluate(None, market, 'cpu', fixed=probs)
    if [x['return_date'] for x in actual['daily']] != [x['return_date'] for x in expected['daily']]:
        raise AssertionError('Dates differ')
    error = float(np.max(abs(np.array([x['net_return'] for x in actual['daily']]) -
                             np.array([x['net_return'] for x in expected['daily']]))))
    if error > 1e-11:
        raise AssertionError('Saved-probability substitution changed original financial trajectory')
    probabilities[name] = probs
    parity[name] = error

draws = []
for seed in range(10):
    row = {'action_seed': seed}
    for name, probs in probabilities.items():
        rng = np.random.default_rng(seed)
        state = market.reset()
        while True:
            wait_for_compute_window()
            _, targets = state
            action = min(int(np.searchsorted(np.cumsum(probs), rng.random())), len(probs)-1)
            state, _, done = market.step(targets[action])
            if done:
                break
        row[name] = metrics_from_infos(market.infos)
    row['Sharpe_gain'] = row['trained']['sharpe_proxy'] - row['initial']['sharpe_proxy']
    draws.append(row)
result = {'scope': 'Reused-development execution sensitivity, not final acceptance. Ten paired action draws share one market history, not ten independent markets. Primary deployment remains deterministic expected-mixture execution.',
          'diagnostic_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          'source_sha256': {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in source.glob('*.py')},
          'data_sha256': hashlib.sha256(data.read_bytes()).hexdigest(),
          'method': 'Constant probabilities from an already fresh-reload-verified native policy; exact deterministic financial parity checked before sampling.',
          'selected_step': step, 'deterministic_daily_parity_errors': parity,
          'draws': draws, 'mean_sampled_Sharpe_gain': float(np.mean([x['Sharpe_gain'] for x in draws])),
          'positive_draws': sum(x['Sharpe_gain'] > 0 for x in draws)}
out.write_text(json.dumps(result, indent=2))
print(json.dumps({k: v for k, v in result.items() if k != 'draws'}, indent=2))
