"""Paired temporal uncertainty for a completed RL run's development gain."""
import argparse,json
from pathlib import Path
import numpy as np

def sharpe(x):return np.sqrt(252)*x.mean()/max(x.std(),1e-12)
def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--replicates',type=int,default=2000);a=p.parse_args()
    run=Path(a.run);selection=json.loads((run/'selection.json').read_text());step=selection['selected']['step']
    initial=json.loads((run/'development_0.json').read_text());trained=json.loads((run/f'development_{step}.json').read_text())
    dates=[r['return_date'] for r in initial['daily']]
    if dates!=[r['return_date'] for r in trained['daily']]:raise ValueError('Unpaired daily dates')
    x=np.array([r['net_return'] for r in initial['daily']]);y=np.array([r['net_return'] for r in trained['daily']])
    rng=np.random.default_rng(20261107);intervals={}
    for block in (5,20,60):
        draws=[]
        for _ in range(a.replicates):
            starts=rng.integers(0,len(x),size=(len(x)+block-1)//block)
            indices=((starts[:,None]+np.arange(block))%len(x)).ravel()[:len(x)]
            draws.append(sharpe(y[indices])-sharpe(x[indices]))
        intervals[str(block)]=np.quantile(draws,[.025,.975]).tolist()
    result={'role':'Reused development comparison, not untouched-test evidence','selected_step':step,
        'dates':{'first':dates[0],'last':dates[-1],'sessions':len(dates)},
        'initial':initial['metrics'],'trained':trained['metrics'],
        'delta_Sharpe':sharpe(y)-sharpe(x),
        'return_change_pp':100*(trained['metrics']['return_total']-initial['metrics']['return_total']),
        'drawdown_change_pp':100*(trained['metrics']['max_drawdown']-initial['metrics']['max_drawdown']),
        'paired_circular_block_95_intervals':intervals,'replicates':a.replicates,
        'scope':'Single optimization seed; paired realized-return resampling, no policy retraining. Development selection/search and multiple comparisons not corrected.'}
    target=run/'learning_comparison.json'
    if target.exists():raise FileExistsError(target)
    target.write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k not in ('initial','trained')},indent=2))

if __name__=='__main__':main()
