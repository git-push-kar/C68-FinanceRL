"""Portable CPU replay of the already consumed final period, with artifact verification."""
import hashlib
import json
import os
from pathlib import Path
import sys
import numpy as np

root = Path(__file__).resolve().parent
manifest = json.loads((root/'BUNDLE_MANIFEST.json').read_text())
for name, expected in manifest['sha256'].items():
    path = (root/name).resolve()
    if not path.is_relative_to(root):raise ValueError('Invalid manifest path')
    with path.open('rb') as f:
        if hashlib.file_digest(f,'sha256').hexdigest()!=expected:raise ValueError('Changed bundle file: '+name)
os.environ.update(CUDA_VISIBLE_DEVICES='-1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
sys.path.insert(0,str(root/'runtime_source'))
import execution_policy
execution_policy.wait_for_compute_window=lambda:None
execution_policy.quiet_now=lambda moment=None:False
from candidate_market import Market,evaluate
import torch
torch.set_num_threads(2)
contract=json.loads((root/'final_evaluation/contract.json').read_text())
market=Market(str(root/'native_final_data_20261116/five_asset_final_with_warmup.csv'), '2026-02-02','2026-10-30')
errors=[]
for entry in contract['entries']:
    expected=json.loads((root/'final_evaluation'/f'seed{entry["seed"]}.json').read_text())
    for key in ('initial','trained'):
        actual=evaluate(None,market,'cpu',fixed=entry[key]['probabilities'])
        if [x['return_date'] for x in actual['daily']] != [x['return_date'] for x in expected[key]['daily']]:raise AssertionError('Dates changed')
        error=max(abs(x['net_return']-y['net_return']) for x,y in zip(actual['daily'],expected[key]['daily']))
        if error>1e-11:raise AssertionError('Replay differs from frozen result')
        errors.append({'seed':entry['seed'],'policy':key,'max_daily_return_error':error})
print(json.dumps({'status':'passed','bundle_hashes_verified':True,'replay':errors,
                  'scope':'Exact replay of already consumed data; not a new test.'},indent=2))
