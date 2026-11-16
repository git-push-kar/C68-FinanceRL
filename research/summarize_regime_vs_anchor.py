import json
from pathlib import Path
import numpy as np
from compare_rl_seeds import sharpe
R=Path(__file__).parent;rows=[];xs=[];ys=[];dates=None
for seed in (42,123,456):
 r=R/f'campaign_runs/nifty_regime_native_seed{seed}_64'
 s=json.loads((r/'selection.json').read_text());c=s['selected'];d=json.loads((r/f'development_{c["step"]}.json').read_text());a=json.loads((r/'static_anchor_development.json').read_text())
 ds=[x['return_date'] for x in d['daily']]
 if ds!=[x['return_date'] for x in a['daily']] or (dates is not None and dates!=ds):raise AssertionError('Unpaired dates')
 dates=ds;xs.append([x['net_return'] for x in a['daily']]);ys.append([x['net_return'] for x in d['daily']])
 rows.append({'seed':seed,'step':c['step'],'matched_initial':s['initial']['development'],'trained':d['metrics'],'static_anchor':a['metrics']})
x=np.asarray(xs);y=np.asarray(ys);g=sharpe(y)-sharpe(x);rng=np.random.default_rng(20261114);draws=[];n=x.shape[1]
for _ in range(5000):
 start=rng.integers(0,n,size=(n+19)//20);idx=((start[:,None]+np.arange(20))%n).ravel()[:n];delta=sharpe(y[:,idx])-sharpe(x[:,idx]);draws.append(delta[rng.integers(0,3,3)].mean())
result={'scope':'Reused development, adaptive method search; no held-out acceptance','runs':rows,'trained_minus_static_anchor_Sharpe':g.tolist(),'mean_gain':float(g.mean()),'paired_20day_seed_95_interval':np.quantile(draws,[.025,.975]).tolist()}
(R/'reports/regime_vs_static_anchor.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='runs'},indent=2))
