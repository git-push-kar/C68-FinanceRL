"""Fixed checkpoints, altered fees only; no training or reselection."""
import os,sys,json,functools
from pathlib import Path
R=Path(__file__).parent;os.environ['CUDA_VISIBLE_DEVICES']='-1';sys.path.insert(0,str(R/'rl_candidate_static_native'));os.chdir(R/'rl_candidate_static_native')
import numpy as np
import torch
import candidate_market,exposure_market
from cash_execution import execute
torch.set_num_threads(2);out=R/'reports/static_cost_sensitivity.json'
if out.exists():raise FileExistsError(out)
rows=[]
for fold,start,end in [('f1','2021-01-01','2021-12-31'),('f2','2022-01-01','2022-12-31'),('native','2023-01-01','2024-12-31')]:
 for seed in (42,123,456):
  name=f'nifty_static_{fold}_seed{seed}_64';r=R/'campaign_runs'/name;s=json.loads((r/'selection.json').read_text())
  initial=json.loads((r/'development_0.json').read_text());trained=json.loads((r/f'development_{s["selected"]["step"]}.json').read_text())
  for fee in (0.,5.,10.,25.):
   candidate_market.execute=exposure_market.execute=functools.partial(execute,fee_bps=fee)
   m=candidate_market.Market('../data/nifty50_prices.csv',start,end);pair={}
   for label,reference in [('initial',initial),('trained',trained)]:
    actual=candidate_market.evaluate(None,m,'cpu',fixed=np.array(reference['probabilities'][0]));pair[label]=actual['metrics']
    if fee==10.:
     error=float(np.max(abs(np.array([i['net_return'] for i in actual['daily']])-np.array([i['net_return'] for i in reference['daily']]))))
     if error>1e-12:raise AssertionError('Main fee no longer reproduces original')
   rows.append({'fold':fold,'seed':seed,'fee_bps_full_turnover':fee,**pair,'Sharpe_gain':pair['trained']['sharpe_proxy']-pair['initial']['sharpe_proxy']})
  print(name,'done',flush=True)
out.write_text(json.dumps({'scope':'Reused development; fixed10bp-selected checkpoints, no cost-based reselection. No reserved data.','rows':rows},indent=2))
