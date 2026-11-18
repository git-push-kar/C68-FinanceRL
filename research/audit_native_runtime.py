"""GPU gate for the native finance router. Run only when the GPU queue is free."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import numpy as np
import torch

p = argparse.ArgumentParser()
for name in ('run', 'source', 'data', 'start', 'end', 'output'):
    p.add_argument('--'+name, required=True)
a = p.parse_args()
if a.end > '2024-12-31':
    raise ValueError('Development-only runtime gate')
run, source, data, output = (Path(x).resolve() for x in (a.run, a.source, a.data, a.output))
output.mkdir(parents=True, exist_ok=False)
sys.path.insert(0, str(source))
os.chdir(source)
from candidate_market import Market
from text_policy import TextPolicy
from shared_backbone import SharedBackbone, DomainRouter
from native_finance_runtime import NativeFinanceSpecialist
from execution_policy import configure_finance_precision, wait_for_compute_window

def read(path):
    return json.loads(path.read_text())

wait_for_compute_window()
configure_finance_precision()
torch.set_num_threads(2)
assert read(run/'reload_audit.json')['status'] == 'passed'
if hashlib.sha256(data.read_bytes()).hexdigest() != read(run/'protocol.json')['dataset_sha256']:
    raise ValueError('Runtime gate data differs from the saved training/evaluation contract')
selection = read(run/'selection.json')['selected']
reference = read(run/f'development_{selection["step"]}.json')
initial = read(run/'development_0.json')
market = Market(str(data), a.start, a.end)
obs, _ = market.reset()
env = market.env
shared = SharedBackbone.load('cuda', 'fp32')
specialist = NativeFinanceSpecialist(selection['package'], TextPolicy, shared)
router = DomainRouter(shared)
specialist.register(router)
payload = {'tickers': env.tickers, 'trading_step': 0,
           'execution_portfolio': [*env.weights.tolist(), float(env.cash_weight)],
           'signal_lag_closes': 1, 'signal_close_date': str(env.dates[env.index-1])[:10],
           'execution_close_date': str(env.dates[env.index])[:10],
           'observation': {key: value.tolist() for key, value in obs.items()}}
prediction = router.predict('finance', payload)
error = float(np.max(abs(np.asarray(prediction['probabilities']) - np.asarray(reference['probabilities'][0]))))
if error > 1e-7:
    raise AssertionError('Router differs from selected research policy')
with shared.model.disable_adapter():
    disabled = router.predict('finance', payload)
base_error = float(np.max(abs(np.asarray(disabled['probabilities']) - np.asarray(initial['probabilities'][0]))))
if base_error > 1e-7:
    raise AssertionError('Router disabled adapter differs from exact initial policy')
restored = router.predict('finance', payload)
if restored != prediction:
    raise AssertionError('Adapter disable/restore changed routed prediction')
market.step(np.asarray(prediction['weights']), max_days=1)
if market.infos[0]['return_date'] != reference['daily'][0]['return_date']:
    raise AssertionError('Gate dates differ from saved development')
daily_error = abs(market.infos[0]['net_return'] - reference['daily'][0]['net_return'])
if daily_error > 1e-11:
    raise AssertionError('Routed allocation changes executed first-day net return')
holdings = [*env.weights.tolist(), float(env.cash_weight)]
hold = router.predict('finance', {'tickers': env.tickers, 'trading_step': 1,
                                  'execution_portfolio': holdings})
if hold['rebalance_due'] or hold['weights'] != holdings[:-1] or hold['cash_weight'] != holdings[-1]:
    raise AssertionError('Off-schedule routing changed actual holdings')
result = {'status': 'passed', 'run': str(run), 'selected_step': selection['step'],
          'same_shared_instance': specialist.policy.shared is router.shared,
          'selected_probability_error': error, 'disabled_probability_error': base_error,
          'first_day_net_return_error': daily_error, 'off_schedule_holds_actual_positions': True,
          'scope': 'Actual selected native finance adapter and shared router; other trained domains not tested.',
          'test2026_accessed': False,
          'runtime_sha256': hashlib.sha256(Path(__file__).with_name('native_finance_runtime.py').read_bytes()).hexdigest()}
for name, value in [('request', payload), ('prediction', prediction), ('result', result)]:
    (output/f'{name}.json').write_text(json.dumps(value, indent=2))
print(json.dumps(result, indent=2))
