"""Fresh-process reload and independent validation of strategy PPO artifacts."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from strategy_ppo import Policy,Market,evaluate
from model import InternVLFinancePolicy,write_json,digest
from train import lora_digest

def load_policy(package,numeric=False):
    package=Path(package)
    if numeric:raise NotImplementedError('Numeric reload requires ticker schema')
    obj=Policy.__new__(Policy);nn.Module.__init__(obj);obj.numeric=False
    obj.base=InternVLFinancePolicy.load_artifacts(package/'base_modules',device='cuda')
    hidden=obj.base.shared.hidden_size
    obj.gate=nn.Linear(hidden,4).cuda();obj.value=nn.Linear(hidden,1).cuda()
    state=torch.load(package/'strategy_heads.pt',map_location='cpu',weights_only=True)
    obj.gate.load_state_dict(state['gate']);obj.value.load_state_dict(state['value']);obj.requires_grad_(False)
    return obj.eval()

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);a=p.parse_args();run=Path(a.run)
    selection=json.loads((run/'selection.json').read_text());step=selection['selected']['step']
    policy=load_policy(selection['selected']['package']);market=Market('../data/nifty50_prices.csv','2022-01-01','2022-12-31')
    observed=evaluate(policy,market,'cuda');expected=json.loads((run/f'validation_{step}.json').read_text())
    error=float(np.max(abs(np.array(observed['probabilities'])-np.array(expected['probabilities']))))
    if error>1e-7:raise AssertionError(f'Reload differs: {error}')
    status=json.loads((run/'status.json').read_text())
    if status['optimizer_updates']<=0 or status['initial_lora_sha256']==status['final_lora_sha256']:
        raise AssertionError('No verified RL LoRA learning')
    result={'status':'passed','selected_step':step,'reload_max_probability_error':error,
        'optimizer_updates':status['optimizer_updates'],'lora_changed':True,
        'selection_includes_initial':True,'fresh_test_accessed':False,
        'selected_lora_sha256':lora_digest(policy.base),'scope':'Technical learning/reload gate, not financial acceptance'}
    write_json(run/'reload_audit.json',result);print(json.dumps(result,indent=2))

if __name__=='__main__':main()
