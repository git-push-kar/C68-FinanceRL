"""Read-only development diagnosis; never selects by development outcomes."""
import json,sys,os
from pathlib import Path
ROOT=Path(__file__).parent
os.environ['CUDA_VISIBLE_DEVICES']='-1'
sys.path.insert(0,str(ROOT/'rl_candidate_trial'))
os.chdir(ROOT/'rl_candidate_trial')
import torch
from candidate_market import Market,evaluate,NAMES
from numeric_policy import NumericPolicy
torch.set_num_threads(2)
out=ROOT/'reports/candidate_recovery_diagnostic_20261114.json'
if out.exists():raise FileExistsError(out)
rows={}
for split,start,end in [('train','2019-01-01','2021-12-31'),('validation','2022-01-01','2022-12-31'),('development','2023-01-01','2024-12-31')]:
 market=Market('../data/nifty50_prices.csv',start,end);policy=NumericPolicy(market.env.tickers)
 controls={name:evaluate(policy,market,'cpu',fixed=i)['metrics'] for i,name in enumerate(NAMES)}
 controls['uniform']=evaluate(policy,market,'cpu')['metrics'];trained={}
 for seed in (42,123,456):
  run=ROOT/f'campaign_runs/nifty_candidate_direct_seed{seed}_64'
  p=NumericPolicy(market.env.tickers,run/'checkpoint_64')
  trained[str(seed)]=evaluate(p,market,'cpu')['metrics']
 rows[split]={'controls':controls,'fixed_endpoint64':trained}
result={'scope':'Diagnostic of fixed endpoint and all declared controls, no new selection or reserved test','splits':rows}
out.write_text(json.dumps(result,indent=2))
for split,row in rows.items():print(split,{k:round(v['sharpe_proxy'],5) for k,v in row['controls'].items()},'trained64',{k:round(v['sharpe_proxy'],5) for k,v in row['fixed_endpoint64'].items()})
