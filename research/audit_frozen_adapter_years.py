import os,sys,json,hashlib
from pathlib import Path
R=Path.cwd(); B=R/'deliverables/finance_rl_native_final_20261116'
sys.path.insert(0,str(B/'runtime_source'))
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import execution_policy
execution_policy.wait_for_compute_window=lambda:None
from candidate_market import Market,evaluate
import torch
torch.set_num_threads(2)
out=R/'additional_frozen_adapter_audit_20261116';out.mkdir(exist_ok=False)
c=json.loads((B/'final_evaluation/contract.json').read_text());entry=c['entries'][0]
protocol={'scope':'Additional historical robustness audit; no retraining or checkpoint selection. 2022-24 development reused; 2025 history accessed in previous project campaigns, not independent untouched test. Annual portfolios reset to cash. Same fixed five ETFs, lagged signals, fees and five-session execution.','policies':{'initial':entry['initial']['probabilities'],'trained_seed42':entry['trained']['probabilities'],'equal_weight':0,'minimum_variance':4},'years':[2022,2023,2024,2025]}
(out/'protocol.json').write_text(json.dumps(protocol,indent=2))
rows=[]
for year in protocol['years']:
 data=B/'multiasset_data_20261115/five_asset_development_through2024.csv' if year<2025 else B/'native_final_data_20261116/five_asset_final_with_warmup.csv'
 m=Market(str(data),f'{year}-01-01',f'{year}-12-31')
 for name,p in protocol['policies'].items():
  result=evaluate(None,m,'cpu',fixed=p)
  (out/f'{year}_{name}.json').write_text(json.dumps(result,indent=2))
  row={'year':year,'policy':name,**result['metrics']};rows.append(row)
  print(year,name,round(row['return_total']*100,3),round(row['sharpe_proxy'],3),flush=True)
(out/'summary.json').write_text(json.dumps(rows,indent=2))
