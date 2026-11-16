"""Matched direct-utility refinement of the independently trained small Transformer.

This is a labelled adaptation of direct Sharpe training, not an exact replication
of a paper's architecture, assets or experimental protocol.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time
import numpy as np
import torch

from env import PortfolioEnv, FinanceRewardConfig
from execution_policy import wait_for_compute_window
from model import digest, write_json
from numeric_covariance_control import NumericCovariancePolicy
from train import seed_everything
from train_covariance_teacher import training_examples, fidelity, evaluate
from train_utility import rollout_loss
from warmup_finance import label_indices
from evaluate_sensitivity import lagged_episode
from metrics import metrics_from_infos


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent-run',required=True);p.add_argument('--output',required=True)
    p.add_argument('--data',default='../data/nifty50_prices.csv');p.add_argument('--seed',type=int,default=42)
    p.add_argument('--updates',type=int,default=128);p.add_argument('--horizon',type=int,default=126)
    p.add_argument('--windows-per-update',type=int,default=4)
    p.add_argument('--anchor-coefficient',type=float,default=5.)
    p.add_argument('--learning-rate',type=float,default=1e-4)
    p.add_argument('--signal-lag-closes',type=int,choices=(0,1),default=0)
    args=p.parse_args();output=Path(args.output);output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1);seed_everything(args.seed)
    folder=Path(args.parent_run);selection=json.loads((folder/'selection.json').read_text())
    parent=folder/f"checkpoint_{selection['selected']['step']}.pt"
    reward=FinanceRewardConfig(rebalance_interval=21)
    env=PortfolioEnv(args.data,end_date='2022-12-31',covariance_window=252,reward_config=reward);env.reset()
    _,holdout,boundary=label_indices(len(env.prices),252,0)
    starts=np.arange(252+args.signal_lag_closes,boundary-args.horizon)
    if len(starts)<2:raise ValueError('Training windows do not fit before chronological holdout')
    assert starts[-1]+args.horizon<boundary
    internal=PortfolioEnv(args.data,start_date=str(env.dates[boundary])[:10],end_date='2022-12-31',
        covariance_window=252,reward_config=reward)
    development=PortfolioEnv(args.data,start_date='2023-01-01',end_date='2024-12-31',
        covariance_window=252,reward_config=reward)
    probe=training_examples(env,holdout[np.linspace(0,len(holdout)-1,32,dtype=int)])
    policy=NumericCovariancePolicy(env.n_assets,attention=True)
    policy.load_state_dict(torch.load(parent,map_location='cpu',weights_only=True),strict=True)
    optimizer=torch.optim.AdamW(policy.parameters(),lr=args.learning_rate,weight_decay=1e-4)
    metadata={'algorithm':'small_causal_transformer_direct_Sharpe_with_teacher_anchor',
        'train_end':'2022-12-31','seed':args.seed,'arguments':vars(args),
        'parent_sha256':digest(parent),'parent_weights':str(parent.resolve()),
        'source_sha256':{f.name:digest(f) for f in Path(__file__).parent.glob('*.py')},
        'dataset_sha256':digest(args.data),'last_training_reward_date':str(env.dates[starts[-1]+args.horizon])[:10],
        'selection':'pre-2023 holdout Sharpe minus anchor_coefficient times teacher KL; includes unchanged parent',
        'test2025_accessed':False,'attention':True,'parameters':sum(p.numel() for p in policy.parameters())}
    write_json(output/'run.json',metadata);candidates=[]
    def evaluate_execution(environment,path):
        if not args.signal_lag_closes:return evaluate(policy,environment,'cpu',path)
        infos=lagged_episode(policy,environment,'cpu',1);metrics=metrics_from_infos(infos)
        write_json(path,{'metrics':metrics,'daily':infos});return metrics
    def assess(step,weights):
        teacher=fidelity(policy,probe,'cpu')
        hold=evaluate_execution(internal,output/f'holdout_{step}.json')
        dev=evaluate_execution(development,output/f'development_{step}.json')
        row={'step':step,'weights':str(weights.resolve()),'fidelity':teacher,'internal_metrics':hold,'metrics':dev,
            'selection_utility':hold['sharpe_proxy']-args.anchor_coefficient*teacher['teacher_KL']}
        candidates.append(row);write_json(output/'candidates.json',candidates)
        print('ASSESSMENT',json.dumps(row),flush=True)
    assess(0,parent);rng=np.random.default_rng(args.seed);logs=[];started=time.monotonic()
    for update in range(1,args.updates+1):
        optimizer.zero_grad(set_to_none=True);samples=[]
        for start in rng.choice(starts,args.windows_per_update):
            wait_for_compute_window()
            loss,details=rollout_loss(policy,env,int(start),args.horizon,args.anchor_coefficient,device='cpu',signal_lag=args.signal_lag_closes)
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite utility objective')
            (loss/args.windows_per_update).backward();samples.append(details)
        norm=torch.nn.utils.clip_grad_norm_(policy.parameters(),1.,error_if_nonfinite=True);optimizer.step()
        if update==1 or update%8==0:
            row={'updates':update,'sample_mean_sharpe':float(np.mean([s['sharpe'] for s in samples])),
                'sample_mean_anchor_KL':float(np.mean([s['anchor_KL'] for s in samples])),
                'grad_norm':float(norm),'seconds':time.monotonic()-started}
            logs.append(row);write_json(output/'losses.json',logs);print(json.dumps(row),flush=True)
        if update%32==0 or update==args.updates:
            weights=output/f'checkpoint_{update}.pt';torch.save(policy.state_dict(),weights);assess(update,weights)
    selected=max(candidates,key=lambda row:(row['selection_utility'],-row['step']))
    write_json(output/'selection.json',{'selected':selected,'initial':candidates[0],
        'criterion':metadata['selection'],'all_candidates':candidates,'seed':args.seed,'attention':True,
        'role':'matched objective numeric Transformer control; no 2025 outcomes'})
    write_json(output/'status.json',{'status':'completed','test2025_accessed':False,'refinement_selected':selected['step']>0})


if __name__=='__main__':main()
