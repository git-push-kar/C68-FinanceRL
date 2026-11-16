"""Amortize causal risk minimization directly, rather than imitate weight KL.

No future returns or optimizer-produced training labels enter the loss. Exact
minimum-variance targets are used only for holdout diagnostics/comparisons.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time
import numpy as np
import torch
from config import ModelConfig
from differentiable_portfolio import capped_weights
from env import PortfolioEnv,FinanceRewardConfig
from execution_policy import wait_for_compute_window
from model import InternVLFinancePolicy,digest,write_json
from numeric_covariance_control import NumericCovariancePolicy
from train import seed_everything,lora_digest
from train_covariance_teacher import training_examples,fidelity,evaluate
from warmup_finance import label_indices


def relative_risk_loss(scores,covariance,cap=.3):
    weights=torch.stack([capped_weights(row,cap) for row in scores])
    cov=covariance.to(torch.float64)
    variance=torch.einsum('bi,bij,bj->b',weights,cov,weights)
    uniform_variance=cov.mean((-1,-2))
    return (variance.clamp_min(1e-16)/uniform_variance.clamp_min(1e-16)).log().mean()


def observations_at(env,indices):
    observations=[]
    for index in indices:
        env.index=int(index);env.weights=np.full(env.n_assets,1/env.n_assets)
        env.cash_weight=0.;env.equity=env.peak_equity=env.initial_cash;env.returns=[]
        observations.append(env._observation())
    return {k:torch.as_tensor(np.stack([o[k] for o in observations]),dtype=torch.float32)
            for k in observations[0]}


@torch.no_grad()
def risk_fidelity(policy,probe,device):
    obs,targets=probe;ratios=[];regrets=[]
    for i in range(len(targets)):
        scores,_=policy({k:v[i:i+1].to(device) for k,v in obs.items()})
        w=capped_weights(scores[0],.3).cpu().numpy();cov=obs['covariance'][i].double().numpy()
        target=targets[i].double().numpy();var=float(w@cov@w)
        ratios.append(var/float(cov.mean()));regrets.append(var/float(target@cov@target))
    return {**fidelity(policy,probe,device),'mean_log_relative_variance':float(np.log(ratios).mean()),
        'mean_variance_to_uniform':float(np.mean(ratios)),'mean_variance_to_exact_optimizer':float(np.mean(regrets)),
        'mean_volatility_to_exact_optimizer':float(np.sqrt(regrets).mean())}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True)
    p.add_argument('--data',default='../data/nifty50_prices.csv');p.add_argument('--seed',type=int,default=42)
    p.add_argument('--steps',type=int,default=512);p.add_argument('--numeric',action='store_true')
    p.add_argument('--no-lora-train',action='store_true');args=p.parse_args()
    if args.numeric and args.no_lora_train:raise ValueError('Numeric model has no LoRA')
    output=Path(args.output);output.mkdir(parents=True,exist_ok=False)
    seed_everything(args.seed);torch.set_num_threads(1 if args.numeric else 2)
    reward=FinanceRewardConfig(rebalance_interval=21)
    env=PortfolioEnv(args.data,end_date='2022-12-31',covariance_window=252,reward_config=reward);env.reset()
    train,holdout,_=label_indices(len(env.prices),252,0)
    probe=training_examples(env,holdout[np.linspace(0,len(holdout)-1,32,dtype=int)])
    development=PortfolioEnv(args.data,start_date='2023-01-01',end_date='2024-12-31',
        covariance_window=252,reward_config=reward)
    cfg=ModelConfig(dtype='fp32',actor_init_scale=.01,score_scale=1.,use_asset_ids=False,
        normalize_market=True,covariance_window=252)
    device='cpu' if args.numeric else 'cuda'
    policy=NumericCovariancePolicy(env.n_assets,attention=True) if args.numeric else InternVLFinancePolicy(cfg,env.tickers,device=device)
    if args.numeric:
        optimizer=torch.optim.AdamW(policy.parameters(),lr=3e-4,weight_decay=1e-4)
    else:
        with torch.no_grad():policy.critic.weight.zero_();policy.critic.bias.zero_()
        policy.critic.requires_grad_(False);policy.log_std.requires_grad_(False)
        lora=[p for n,p in policy.shared.named_parameters() if p.requires_grad and 'lora_' in n]
        if args.no_lora_train:
            for parameter in lora:parameter.requires_grad_(False)
        optimizer=torch.optim.AdamW([
            {'params':policy.actor.parameters(),'lr':5e-4,'weight_decay':0.},
            {'params':policy.encoders.parameters(),'lr':1e-4,'weight_decay':1e-4},
            {'params':[] if args.no_lora_train else lora,'lr':2e-5,'weight_decay':0.}])
    metadata={'algorithm':'causal_log_relative_portfolio_variance_minimization_not_RL',
        'train_end':'2022-12-31','seed':args.seed,'arguments':vars(args),'covariance_window':252,
        'execution_interval':21,'batch_size':4,'source_sha256':{f.name:digest(f) for f in Path(__file__).parent.glob('*.py')},
        'dataset_sha256':digest(args.data),'selection':'lowest pre-2023 mean log relative portfolio variance; earlier checkpoint on tie',
        'training_labels':'none; known covariance supplies the differentiable risk objective',
        'initial_lora_sha256':None if args.numeric else lora_digest(policy),
        'validation_role':'reused development, not final test','test2025_accessed':False,
        'optimizer':({'all_lr':3e-4} if args.numeric else {'actor_lr':5e-4,'encoder_lr':1e-4,'lora_lr':0. if args.no_lora_train else 2e-5}),
        'no_lora_train':args.no_lora_train,'numeric':args.numeric}
    metadata['numerical_backend']={'float32_matmul_precision':torch.get_float32_matmul_precision(),
        'cuda_matmul_allow_tf32':torch.backends.cuda.matmul.allow_tf32,
        'cudnn_allow_tf32':torch.backends.cudnn.allow_tf32}
    if not args.numeric and metadata['numerical_backend']!={'float32_matmul_precision':'highest',
            'cuda_matmul_allow_tf32':False,'cudnn_allow_tf32':False}:
        raise RuntimeError('Strict FP32 finance training requires both TF32 permissions disabled')
    write_json(output/'run.json',metadata)
    initial={'fidelity':risk_fidelity(policy,probe,device),
        'metrics':evaluate(policy,development,device,output/'initial_daily.json')}
    write_json(output/'initial.json',initial)
    if not args.numeric:policy.save_artifacts(output/'initial',{'algorithm':'risk_initialization'},asdict(reward),
        {**metadata,'optimizer_updates':0,'learning_verified':False})
    rng=np.random.default_rng(args.seed);candidates=[];logs=[];started=time.monotonic()
    for step in range(1,args.steps+1):
        wait_for_compute_window();obs=observations_at(env,rng.choice(train,4))
        obs={k:v.to(device) for k,v in obs.items()};scores,_=policy(obs)
        loss=relative_risk_loss(scores,obs['covariance'])
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite relative risk')
        optimizer.zero_grad(set_to_none=True);loss.backward()
        norm=torch.nn.utils.clip_grad_norm_([p for p in policy.parameters() if p.requires_grad],1.,error_if_nonfinite=True);optimizer.step()
        if step==1 or step%32==0:
            row={'step':step,'log_relative_risk':float(loss.detach()),'grad_norm':float(norm),
                'seconds':time.monotonic()-started,'peak_allocated_GiB':0. if args.numeric else torch.cuda.max_memory_allocated()/2**30}
            logs.append(row);write_json(output/'losses.json',logs);print(json.dumps(row),flush=True)
        if step%128==0 or step==args.steps:
            package=output/f'checkpoint_{step}'
            if args.numeric:
                weights=output/f'checkpoint_{step}.pt';torch.save(policy.state_dict(),weights)
            else:
                current=lora_digest(policy)
                policy.save_artifacts(package,{'algorithm':metadata['algorithm']},asdict(reward),
                    {**metadata,'steps':step,'optimizer_updates':step,'final_lora_sha256':current,
                     'learning_verified':current!=metadata['initial_lora_sha256']})
            row={'step':step,'fidelity':risk_fidelity(policy,probe,device),
                'metrics':evaluate(policy,development,device,output/f'development_{step}.json')}
            row.update({'weights':str(weights.resolve())} if args.numeric else {'package':str(package)})
            candidates.append(row);write_json(output/'candidates.json',candidates);print('CHECKPOINT',json.dumps(row),flush=True)
    selected=min(candidates,key=lambda row:(row['fidelity']['mean_log_relative_variance'],row['step']))
    write_json(output/'selection.json',{'criterion':metadata['selection'],'selected':selected,'initial':initial,
        'attention':args.numeric,'seed':args.seed})
    write_json(output/'status.json',{'status':'completed','test2025_accessed':False,'financial_generalization_proven':False})


if __name__=='__main__':main()
