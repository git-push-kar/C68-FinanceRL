"""Actual selected-model state sensitivity on validation cases; no training."""
import copy,json,argparse
from pathlib import Path
import numpy as np
from peft import set_peft_model_state_dict
from safetensors.torch import load_file
from policy import NativePolicy

p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--base',required=True);a=p.parse_args();run=Path(a.run)
selection=json.loads((run/'selection.json').read_text());step=selection['step']
records=json.loads((run/f'validation_step{step}_capital100000.json').read_text())['decisions']
model=NativePolicy(a.base);set_peft_model_state_dict(model.model,load_file(str(run/f'checkpoint_{step}/finance_lora/finance/adapter_model.safetensors')),adapter_name='finance')
results=[]
for index in range(0,len(records),2):
 obs=records[index]['observation'];actions,receipt=model.choose(obs)
 item={'date':obs['asof'],'original_actions':actions.tolist(),'counterfactuals':{}}
 for kind in ['neutral_market_features','reversed_trailing_returns']:
  altered=copy.deepcopy(obs)
  for stock in altered['stocks']:
   for key in ['return5','return20','return60','return120']:stock[key]=0. if kind=='neutral_market_features' else -stock[key]
   if kind=='neutral_market_features':stock.update(vol20=.2,drawdown60=0.,volume_ratio=1.)
  changed,r=model.choose(altered)
  item['counterfactuals'][kind]={'actions':changed.tolist(),'changed_action_count':int(np.sum(changed!=actions)),'max_probability_change':float(np.max(abs(np.asarray(receipt['probabilities'])-np.asarray(r['probabilities']))))}
 results.append(item)
out={'scope':'Actual model validation-state diagnostic; synthetic feature counterfactuals are not performance/backtest claims. Cash, prices, holdings and identities are unchanged.','selected_step':step,'cases':results}
(run/'validation_feature_sensitivity.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out,indent=2))
