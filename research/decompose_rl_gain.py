"""Exact realized log-wealth decomposition into market and trading-cost terms."""
import argparse,json
from pathlib import Path
import numpy as np

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--budget',type=int)
    a=p.parse_args();run=Path(a.run)
    if json.loads((run/'status.json').read_text())['status']!='completed':raise ValueError('Run unfinished')
    if a.budget is None:
        step=json.loads((run/'selection.json').read_text())['selected']['step'];suffix='selected'
    else:
        cs=[c for c in json.loads((run/'candidates.json').read_text()) if c['step']<=a.budget]
        if max(c['step'] for c in cs)!=a.budget:raise ValueError('Budget unavailable')
        step=max(cs,key=lambda c:(c['validation']['sharpe_proxy'],-c['step']))['step'];suffix=f'budget{a.budget}'
    records=[json.loads((run/f'development_{s}.json').read_text()) for s in (0,step)]
    if [d['return_date'] for d in records[0]['daily']]!=[d['return_date'] for d in records[1]['daily']]:raise ValueError('Unpaired dates')
    arrays=[]
    for r in records:
        gross=np.array([d['gross_return'] for d in r['daily']]);cost=np.array([d['trading_cost'] for d in r['daily']])
        net=np.array([d['net_return'] for d in r['daily']])
        components=np.stack([np.log1p(gross),np.log1p(-cost),np.log1p(net)])
        if np.max(abs(components[0]+components[1]-components[2]))>1e-12:raise AssertionError('Accounting identity failed')
        arrays.append(components)
    delta=arrays[1]-arrays[0];rng=np.random.default_rng(20261107);n=delta.shape[1];draws=[]
    for _ in range(5000):
        starts=rng.integers(0,n,size=(n+19)//20);idx=((starts[:,None]+np.arange(20))%n).ravel()[:n]
        draws.append(100*delta[:,idx].sum(1))
    ci=np.quantile(draws,[.025,.975],axis=0)
    result={'scope':'Reused development, exact accounting for realized action paths; not a zero-cost counterfactual rerun',
        'selected_step':step,'budget':a.budget,'units':'100 times cumulative log wealth (log-return percentage points)',
        'trained_minus_initial':{key:float(value) for key,value in zip(['gross_market_component','cost_component','net_total'],100*delta.sum(1))},
        'paired_20_day_block_95_intervals':{key:ci[:,i].tolist() for i,key in enumerate(['gross_market_component','cost_component','net_total'])},
        'initial_mean_turnover':records[0]['metrics']['mean_turnover'],'trained_mean_turnover':records[1]['metrics']['mean_turnover'],
        'limitations':'Components sum exactly in log wealth, not in ordinary total-return percentage points. Actions and holdings come from the actual 10bp run. Intervals do not correct development search.'}
    out=run/f'cost_decomposition_{suffix}.json'
    if out.exists():raise FileExistsError(out)
    out.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

if __name__=='__main__':main()
