"""Train the same causal risk task without a pretrained language backbone."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time
import numpy as np
import torch
from torch import nn

from config import ModelConfig
from env import PortfolioEnv,FinanceRewardConfig
from execution_policy import wait_for_compute_window
from model import FinanceEncoders,write_json
from train import seed_everything
from train_covariance_teacher import training_examples,fidelity,evaluate
from warmup_finance import label_indices


class NumericCovariancePolicy(nn.Module):
    def __init__(self,assets,attention=False):
        super().__init__()
        self.cfg=ModelConfig(dtype='fp32',covariance_window=252,use_asset_ids=False,normalize_market=True)
        self.encoders=FinanceEncoders(64,assets,64,use_asset_ids=False,normalize_market=True,covariance_context=True)
        self.transformer=(nn.TransformerEncoder(nn.TransformerEncoderLayer(64,4,dim_feedforward=256,
            dropout=0.,activation='gelu',batch_first=True),num_layers=2,enable_nested_tensor=False)
            if attention else None)
        self.actor=nn.Linear(64,1);self.assets=assets
        with torch.no_grad():self.actor.weight.mul_(.01);self.actor.bias.mul_(.01)

    def forward(self,obs):
        tokens=self.encoders(obs['market'],obs['portfolio'],obs['risk'],obs['covariance'])
        if self.transformer is not None:
            mask=torch.triu(torch.ones(tokens.shape[1],tokens.shape[1],device=tokens.device,dtype=torch.bool),diagonal=1)
            tokens=self.transformer(tokens,mask=mask,is_causal=True)
        queries=tokens[:,2+self.assets:2+2*self.assets]
        return self.actor(queries).squeeze(-1),torch.zeros(len(queries),device=queries.device)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True);p.add_argument('--seed',type=int,default=42)
    p.add_argument('--steps',type=int,default=512);p.add_argument('--data',default='../data/nifty50_prices.csv')
    p.add_argument('--attention',action='store_true',help='Two small causal Transformer blocks allow global cross-asset interaction')
    p.add_argument('--learning-rate',type=float,default=1e-3)
    args=p.parse_args();output=Path(args.output);output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1);seed_everything(args.seed)
    reward=FinanceRewardConfig(rebalance_interval=21)
    env=PortfolioEnv(args.data,end_date='2022-12-31',covariance_window=252,reward_config=reward);env.reset()
    train,holdout,_=label_indices(len(env.prices),252,0)
    probe=training_examples(env,holdout[np.linspace(0,len(holdout)-1,32,dtype=int)])
    validation=PortfolioEnv(args.data,start_date='2023-01-01',end_date='2024-12-31',
        covariance_window=252,reward_config=reward)
    policy=NumericCovariancePolicy(env.n_assets,args.attention)
    optimizer=torch.optim.AdamW(policy.parameters(),lr=args.learning_rate,weight_decay=1e-4)
    initial={'fidelity':fidelity(policy,probe,'cpu'),
        'metrics':evaluate(policy,validation,'cpu',output/'initial_daily.json')}
    write_json(output/'initial.json',initial)
    rng=np.random.default_rng(args.seed);candidates=[];started=time.monotonic()
    for step in range(1,args.steps+1):
        wait_for_compute_window()
        obs,target=training_examples(env,rng.choice(train,4))
        scores,_=policy(obs)
        loss=-(target*torch.log_softmax(scores,-1)).sum(-1).mean()
        optimizer.zero_grad(set_to_none=True);loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(),1.,error_if_nonfinite=True);optimizer.step()
        if step%128==0 or step==args.steps:
            row={'step':step,'fidelity':fidelity(policy,probe,'cpu'),
                'metrics':evaluate(policy,validation,'cpu',output/f'development_{step}.json')}
            torch.save(policy.state_dict(),output/f'checkpoint_{step}.pt')
            candidates.append(row);write_json(output/'candidates.json',candidates)
            print(json.dumps(row),flush=True)
    selected=min(candidates,key=lambda row:(row['fidelity']['teacher_KL'],row['step']))
    write_json(output/'selection.json',{'initial':initial,'selected':selected,
        'criterion':'lowest pre-2023 holdout teacher KL; earlier on tie','seed':args.seed,
        'role':'numeric-only control, same features/teacher/effective batch/512 updates/cadence/costs',
        'parameters':sum(p.numel() for p in policy.parameters()),'lr':args.learning_rate,'attention':args.attention,
        'seconds':time.monotonic()-started,'test2025_accessed':False})


if __name__=='__main__':main()
