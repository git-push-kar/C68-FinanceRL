import argparse,json,sys,os
from pathlib import Path
import numpy as np
R=Path(__file__).parent;a=argparse.ArgumentParser();a.add_argument('--run',required=True);r=Path(a.parse_args().run).resolve();sys.path.insert(0,str(R/'rl_multiasset_dynamic'));os.chdir(R/'rl_multiasset_dynamic')
from candidate_market import Market,evaluate,NAMES
status=json.loads((r/'status.json').read_text())
if status['status']!='completed':raise ValueError('Unfinished run')
s=json.loads((r/'selection.json').read_text())['selected'];v=json.loads((r/f'validation_{s["step"]}.json').read_text());d=json.loads((r/f'development_{s["step"]}.json').read_text());base=json.loads((r/'development_0.json').read_text())
mix=np.asarray(v['probabilities']).mean(0);mix/=mix.sum();m=Market('../multiasset_data_20261115/five_asset_development_through2024.csv','2022-01-01','2024-12-31')
fixed=evaluate(None,m,'cpu',fixed=mix)
if [i['return_date'] for i in d['daily']]!=[i['return_date'] for i in fixed['daily']]:raise AssertionError('Date mismatch')
decomposition={}
for name,p in [('initial',base),('trained',d)]:
 gross=np.array([x['gross_return'] for x in p['daily']]);net=np.array([x['net_return'] for x in p['daily']])
 decomposition[name]={'gross_log_growth':float(np.log1p(gross).sum()),'net_log_growth':float(np.log1p(net).sum()),'cost_log_drag':float((np.log1p(gross)-np.log1p(net)).sum())}
result={'scope':'Development-only explanatory control. Validation-average probabilities do not use development returns. Not causal attribution to timing.','step':s['step'],'validation_average_mixture':mix.tolist(),'development_probability_std':np.asarray(d['probabilities']).std(0).tolist(),'dynamic':d['metrics'],'constant_mixture':fixed['metrics'],'dynamic_minus_constant_Sharpe':d['metrics']['sharpe_proxy']-fixed['metrics']['sharpe_proxy'],'gross_net_cost_decomposition':decomposition}
out=r/'multiasset_adaptation_diagnostic.json'
if out.exists():raise FileExistsError(out)
out.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
