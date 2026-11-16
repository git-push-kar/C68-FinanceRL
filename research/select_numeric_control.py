"""Choose the numeric-control learning rate by validation, never development."""
import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).parent

def main():
    rows=[]
    for lr,label in [(1e-4,None),(3e-4,'0p0003'),(1e-3,'0p001')]:
        for seed in (42,123,456):
            name=f'strategy_ppo_numeric_seed{seed}_256' if label is None else f'strategy_ppo_numeric_lr{label}_seed{seed}_128'
            run=ROOT/'campaign_runs'/name
            if json.loads((run/'status.json').read_text())['status']!='completed':raise ValueError('Control unfinished')
            candidates=[c for c in json.loads((run/'candidates.json').read_text()) if c['step']<=128]
            if max(c['step'] for c in candidates)!=128:raise ValueError('Matched budget unavailable')
            chosen=max(candidates,key=lambda c:(c['validation']['sharpe_proxy'],-c['step']))
            rows.append({'lr':lr,'seed':seed,'run':str(run),'step':chosen['step'],
                'validation_Sharpe':chosen['validation']['sharpe_proxy'],
                'development_Sharpe':chosen['development']['sharpe_proxy'],
                'development_gain':chosen['development']['sharpe_proxy']-candidates[0]['development']['sharpe_proxy']})
    means={lr:float(np.mean([r['validation_Sharpe'] for r in rows if r['lr']==lr])) for lr in (1e-4,3e-4,1e-3)}
    rate=max(means,key=lambda lr:(means[lr],-lr))
    result={'scope':'Reused development calibration; no new test accessed','matched_iterations':128,
        'selection':'Mean validation Sharpe across all three seeds; ties prefer lower learning rate',
        'mean_validation_Sharpe_by_lr':means,'selected_lr':rate,'selected_runs':[r['run'] for r in rows if r['lr']==rate],
        'all_trials':rows,'limitation':'A bounded optimizer-rate calibration, not exhaustive optimization of every baseline.'}
    out=ROOT/'reports/numeric_control_calibration.json'
    if out.exists():raise FileExistsError(out)
    out.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

if __name__=='__main__':main()
