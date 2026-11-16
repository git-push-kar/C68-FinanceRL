"""Fixed-budget endpoint diagnostic, never overrides validation selection."""
import argparse,json,sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).parent;sys.path.insert(0,str(ROOT/'rl_exposure_direct_trial'))
from exposure_market import Market,evaluate
from numeric_policy import NumericPolicy

def annual(daily):
    result={}
    for year in sorted({x['return_date'][:4] for x in daily}):
        rows=[x for x in daily if x['return_date'].startswith(year)];r=np.array([x['net_return'] for x in rows]);equity=np.r_[1.,np.cumprod(1+r)]
        result[year]={'sessions':len(r),'Sharpe':float(np.sqrt(252)*r.mean()/r.std()),
            'return':float(equity[-1]-1),'max_drawdown':float(np.max(1-equity/np.maximum.accumulate(equity)))}
    return result

p=argparse.ArgumentParser();p.add_argument('--run',required=True);a=p.parse_args();run=Path(a.run)
if json.loads((run/'status.json').read_text())['status']!='completed':raise ValueError('Completed run required')
torch.set_num_threads(2);data=ROOT/'sector_data_20261107/sector9_development_through2024.csv'
train=Market(data,'2001-01-01','2018-12-31');initial=NumericPolicy(train.env.tickers,run/'checkpoint_0');trained=NumericPolicy(train.env.tickers,run/'checkpoint_256')
base=evaluate(initial,train,'cpu');last=evaluate(trained,train,'cpu')
result={'scope':'Diagnostic of predeclared 256-update endpoint, NOT the validation-selected policy and NOT a replacement selection rule',
    'selected_step':json.loads((run/'selection.json').read_text())['selected']['step'],
    'training_initial':base['metrics'],'training_endpoint':last['metrics'],'annual':{'training_initial':annual(base['daily']),'training_endpoint':annual(last['daily'])}}
for split in ('validation','development'):
    for label,step in [('initial',0),('endpoint',256)]:
        value=json.loads((run/f'{split}_{step}.json').read_text());result['annual'][f'{split}_{label}']=annual(value['daily'])
out=run/'endpoint_regime_diagnostic.json'
if out.exists():raise FileExistsError(out)
out.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
