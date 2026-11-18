"""Frozen-policy cost and one-close signal-delay development sensitivities."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import numpy as np
import torch
from env import PortfolioEnv,FinanceRewardConfig
from evaluate_policy import run_episode,tensor_obs
from metrics import metrics_from_infos
from model import InternVLFinancePolicy,write_json
from train_covariance_risk import risk_fidelity
from train_covariance_teacher import training_examples
from warmup_finance import label_indices


@torch.no_grad()
def lagged_episode(policy,env,device,lag,observer=None):
    if lag==0:return run_episode(policy,env,device,observer=observer)
    if lag!=1:raise ValueError('This diagnostic implements exactly one close of signal delay')
    was_training=policy.training;policy.eval()
    obs,_=env.reset();start=env.index
    env.index=start-1;previous=env._observation();env.index=start
    infos=[]
    try:
        while True:
            signal_date=str(env.dates[env.index-1])[:10]
            scores,_=policy(tensor_obs(previous,device))
            if observer is not None:observer(env,previous,scores[0].float().cpu().numpy().copy())
            previous=obs
            obs,_,done,truncated,info=env.step(scores[0].float().cpu().numpy().copy())
            info['signal_close_date']=signal_date
            if not signal_date<info['decision_date']:raise RuntimeError('Signal is not lagged')
            infos.append(info)
            if done or truncated:return infos
    finally:policy.train(was_training)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--package',required=True);p.add_argument('--output',required=True)
    p.add_argument('--data',default='../data/nifty50_prices.csv')
    args=p.parse_args();output=Path(args.output);output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2)
    cfg=json.loads((Path(args.package)/'environment_config.json').read_text())
    policy=InternVLFinancePolicy.load_artifacts(args.package,device='cuda')
    training=PortfolioEnv(args.data,end_date='2022-12-31',covariance_window=252,
        reward_config=FinanceRewardConfig(**cfg['reward']));training.reset()
    _,holdout,_=label_indices(len(training.prices),252,0)
    probe=training_examples(training,holdout[np.linspace(0,len(holdout)-1,32,dtype=int)])
    allocation_quality=risk_fidelity(policy,probe,'cuda')
    write_json(output/'pre2023_allocation_quality.json',allocation_quality)
    results={}
    for cost,lag in [(0.,0),(5.,0),(10.,0),(20.,0),(10.,1)]:
        reward=FinanceRewardConfig(**{**cfg['reward'],'transaction_cost_bps':cost})
        env=PortfolioEnv(args.data,start_date='2023-01-01',end_date='2024-12-31',
            covariance_window=cfg.get('covariance_window',0),reward_config=reward)
        infos=lagged_episode(policy,env,'cuda',lag)
        name=f'cost{cost:g}_lag{lag}'
        results[name]=metrics_from_infos(infos)
        write_json(output/(name+'.json'),{'metrics':results[name],'daily':infos})
        print(name,json.dumps(results[name]),flush=True)
    write_json(output/'summary.json',{'role':'frozen-policy development sensitivities',
        'results':results,'pre2023_allocation_quality':allocation_quality,'test2025_accessed':False,
        'delay_definition':'Entire policy observation is delayed one close, including its previous pretrade portfolio/risk state; execution uses actual current holdings. Initial prior-close state is cash.',
        'cost_definition':'bps times traded-stock L1; each cost is a fresh policy simulation, not a fixed-decision replay.',
        'development_reference':'cost10_lag0 reproduces historical development; the final protocol separately fixes cost10 and one-close delay. No model or favorable final-setting selection from this grid.'})


if __name__=='__main__':main()
