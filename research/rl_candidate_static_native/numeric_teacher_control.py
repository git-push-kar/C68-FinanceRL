"""Matched-input GRU control: test whether InternVL adds value to rule fitting."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from torch import nn
from config import ModelConfig
from env import PortfolioEnv, FinanceRewardConfig
from evaluate_policy import run_episode, baseline_episode
from execution_policy import wait_for_compute_window
from metrics import metrics_from_infos
from model import FinanceEncoders
from train import seed_everything
from warmup_finance import examples,label_indices,predict_probe,shuffle_assets


class NumericPolicy(nn.Module):
    def __init__(self,assets):
        super().__init__()
        self.cfg=ModelConfig(dtype='fp32',use_asset_ids=False,normalize_market=True)
        self.encoders=FinanceEncoders(64,assets,64,use_asset_ids=False,normalize_market=True)
        self.actor=nn.Linear(64,1)
        self.assets=assets
        with torch.no_grad():self.actor.weight.mul_(.01);self.actor.bias.mul_(.01)

    def forward(self,obs):
        tokens=self.encoders(obs['market'],obs['portfolio'],obs['risk'])
        states=tokens[:,2+self.assets:2+2*self.assets]
        return self.actor(states).squeeze(-1),torch.zeros(len(states),device=states.device)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True);p.add_argument('--seed',type=int,default=42)
    p.add_argument('--steps',type=int,default=512)
    args=p.parse_args()
    output=Path(args.output);output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1);seed_everything(args.seed)
    env=PortfolioEnv('data/nifty50_prices.csv',end_date='2022-12-31')
    validation=PortfolioEnv('data/nifty50_prices.csv',start_date='2023-01-01',end_date='2024-12-31',
        reward_config=FinanceRewardConfig(rebalance_rate=.1))
    train,holdout,_=label_indices(len(env.prices),30,0)
    probe=examples(env,holdout[np.linspace(0,len(holdout)-1,32,dtype=int)],0,'risk_rule')
    policy=NumericPolicy(env.n_assets)
    optimizer=torch.optim.Adam(policy.parameters(),lr=1e-3)
    rng=np.random.default_rng(args.seed);perm=torch.Generator().manual_seed(args.seed+1)
    result={'seed':args.seed,'steps':args.steps,'role':'numeric-only matched-input causal-teacher control',
        'learning_rate':1e-3,'parameters':sum(x.numel() for x in policy.parameters()),
        'note':'GRU/heads only, not a pretrained adapter. Same teacher, features, training cutoff and portfolio accounting.',
        'initial':{'probe':predict_probe(policy,probe,'cpu'),
                   'metrics':metrics_from_infos(run_episode(policy,validation,'cpu'))}}
    started=time.monotonic()
    for step in range(args.steps):
        wait_for_compute_window()
        obs,target=examples(env,rng.choice(train,4),0,'risk_rule')
        obs,target=shuffle_assets(obs,target,perm)
        scores,_=policy(obs);scores=scores-scores.mean(-1,keepdim=True)
        loss=(scores-target).square().mean()
        optimizer.zero_grad();loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(),1.);optimizer.step()
    result.update(seconds=time.monotonic()-started,
        final={'probe':predict_probe(policy,probe,'cpu'),
               'metrics':metrics_from_infos(run_episode(policy,validation,'cpu'))})
    (output/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    torch.save(policy.state_dict(),output/'numeric_control.pt')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
