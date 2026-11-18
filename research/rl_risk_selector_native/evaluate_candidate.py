"""Fresh-process reload, actual LoRA-off and irrelevant-state diagnostics."""
import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import torch
from torch import nn
from env import PortfolioEnv,FinanceRewardConfig
from evaluate_policy import run_episode
from metrics import metrics_from_infos
from model import InternVLFinancePolicy,digest,write_json


class StateDiagnostic(nn.Module):
    def __init__(self,policy,neutral=False):
        super().__init__();self.policy=policy;self.cfg=policy.cfg;self.neutral=neutral

    def forward(self,obs):
        if self.neutral:
            obs=dict(obs)
            portfolio=torch.zeros_like(obs['portfolio'])
            portfolio[:,:-1]=1/(portfolio.shape[-1]-1)
            obs['portfolio']=portfolio
            obs['risk']=torch.zeros_like(obs['risk'])
        return self.policy(obs)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',required=True);p.add_argument('--output',required=True)
    p.add_argument('--data',default='../data/nifty50_prices.csv')
    args=p.parse_args();output=Path(args.output);output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2)
    selection=json.loads((Path(args.run)/'selection.json').read_text())
    package=Path(selection['selected']['package']).resolve()
    env_cfg=json.loads((package/'environment_config.json').read_text())
    policy=InternVLFinancePolicy.load_artifacts(package,device='cuda')
    report={'package':str(package),'manifest_sha256':digest(package/'manifest.json'),
            'role':'reused development inference dependence and state-distribution diagnostics',
            'test2025_accessed':False,'results':{}}
    for name,neutral,disable in [('full',False,False),('lora_off',False,True),
        ('neutral_state',True,False),('neutral_state_lora_off',True,True)]:
        env=PortfolioEnv(args.data,start_date='2023-01-01',end_date='2024-12-31',
            covariance_window=env_cfg.get('covariance_window',0),
            reward_config=FinanceRewardConfig(**env_cfg['reward']))
        wrapper=StateDiagnostic(policy,neutral)
        with policy.shared.model.disable_adapter() if disable else nullcontext():
            infos=run_episode(wrapper,env,'cuda')
        result={'metrics':metrics_from_infos(infos),'daily':infos}
        write_json(output/(name+'.json'),result);report['results'][name]=result['metrics']
        print(name,json.dumps(result['metrics']),flush=True)
    expected=selection['selected']['metrics']
    actual=report['results']['full']
    if abs(expected['return_total']-actual['return_total'])>1e-6:
        raise RuntimeError('Fresh-process reload did not reproduce selected development result')
    report['reload_development_parity']=True
    report['note']='LoRA-off is an inference ablation, not a separately trained baseline. Neutral-state modes are causal diagnostics: the minimum-variance teacher does not depend on holdings/risk state.'
    write_json(output/'summary.json',report)


if __name__=='__main__':main()
