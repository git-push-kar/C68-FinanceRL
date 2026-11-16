"""Paired temporal and seed uncertainty without treating seeds as new market days."""
import argparse,json
from pathlib import Path
import numpy as np

def sharpe(x):
    return np.sqrt(252)*x.mean(axis=-1)/np.maximum(x.std(axis=-1),1e-12)

def main():
    p=argparse.ArgumentParser();p.add_argument('--runs',nargs='+',required=True)
    p.add_argument('--budget',type=int,required=True);p.add_argument('--output',required=True)
    p.add_argument('--replicates',type=int,default=5000);a=p.parse_args()
    out=Path(a.output)
    if out.exists():raise FileExistsError(out)
    xs=[];ys=[];rows=[];dates=None
    for name in a.runs:
        run=Path(name)
        if json.loads((run/'status.json').read_text())['status']!='completed':raise ValueError('Run incomplete')
        cs=[c for c in json.loads((run/'candidates.json').read_text()) if c['step']<=a.budget]
        if max(c['step'] for c in cs)!=a.budget:raise ValueError('Matched budget unavailable')
        c=max(cs,key=lambda c:(c['validation']['sharpe_proxy'],-c['step']))
        x=json.loads((run/'development_0.json').read_text());y=json.loads((run/f'development_{c["step"]}.json').read_text())
        dx=[r['return_date'] for r in x['daily']];dy=[r['return_date'] for r in y['daily']]
        if dx!=dy or (dates is not None and dates!=dx):raise ValueError('Dates must match across all pairs')
        dates=dx;xs.append([r['net_return'] for r in x['daily']]);ys.append([r['net_return'] for r in y['daily']])
        rows.append({'run':str(run),'selected_step':c['step'],'initial':x['metrics'],'trained':y['metrics']})
    x=np.array(xs);y=np.array(ys);n=x.shape[1];gains=sharpe(y)-sharpe(x)
    rng=np.random.default_rng(20261107);cis={}
    for block in (5,20,60):
        temporal=[];joint=[]
        for _ in range(a.replicates):
            starts=rng.integers(0,n,size=(n+block-1)//block)
            idx=((starts[:,None]+np.arange(block))%n).ravel()[:n]
            delta=sharpe(y[:,idx])-sharpe(x[:,idx])
            temporal.append(delta.mean());joint.append(delta[rng.integers(0,len(delta),size=len(delta))].mean())
        cis[str(block)]={'shared_time_blocks_conditional_on_trained_seeds':np.quantile(temporal,[.025,.975]).tolist(),
                        'shared_time_blocks_and_resampled_seeds':np.quantile(joint,[.025,.975]).tolist()}
    result={'scope':'Reused development, not untouched test; no adjustment for adaptive method search',
        'budget':a.budget,'training_seeds':len(rows),'dates':{'first':dates[0],'last':dates[-1],'sessions':n},
        'individual_Sharpe_gains':gains.tolist(),'mean_Sharpe_gain':float(gains.mean()),
        'median_Sharpe_gain':float(np.median(gains)),'positive_seeds':int((gains>0).sum()),
        'paired_95_intervals':cis,'replicates':a.replicates,'runs':rows,
        'interpretation':'Primary estimate averages per-seed Sharpe improvements, not an ensemble return. One common time-index draw is used across all seeds. Three seeds give limited training-variation resolution; a bootstrap cannot undo development search.'}
    out.write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='runs'},indent=2))

if __name__=='__main__':main()
