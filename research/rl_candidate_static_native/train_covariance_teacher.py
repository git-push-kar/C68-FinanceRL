"""Learn a causal constrained risk optimizer, then evaluate matched controls.

Teacher decisions contain no future labels. This is supervised distillation,
not PPO, market forecasting, or a claim to have invented minimum variance.
"""
import argparse
from contextlib import nullcontext
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch

from classical_panel import min_variance
from config import ModelConfig
from env import PortfolioEnv,FinanceRewardConfig
from evaluate_policy import run_episode
from execution_policy import wait_for_compute_window
from metrics import metrics_from_infos
from model import InternVLFinancePolicy,digest,write_json
from train import seed_everything,lora_digest
from warmup_finance import label_indices


def training_examples(env,indices):
    observations=[];targets=[]
    for index in indices:
        env.index=int(index);env.weights=np.full(env.n_assets,1/env.n_assets)
        env.cash_weight=0.;env.equity=env.peak_equity=env.initial_cash;env.returns=[]
        obs=env._observation()
        observations.append(obs)
        targets.append(min_variance(obs['covariance'].astype(np.float64)))
    return ({k:torch.as_tensor(np.stack([o[k] for o in observations]),dtype=torch.float32)
             for k in observations[0]},torch.as_tensor(np.stack(targets),dtype=torch.float32))


@torch.no_grad()
def fidelity(policy,examples,device):
    observations,targets=examples
    predictions=[]
    for i in range(len(targets)):
        scores,_=policy({k:v[i:i+1].to(device) for k,v in observations.items()})
        predictions.append(torch.softmax(scores,-1).cpu())
    weights=torch.cat(predictions).clamp_min(1e-12)
    kl=(targets*(targets.clamp_min(1e-12).log()-weights.log())).sum(-1)
    return {'teacher_KL':float(kl.mean()),'mean_weight_L1':float((weights-targets).abs().sum(-1).mean()),
            'dates':len(targets)}


@torch.no_grad()
def evaluate(policy,env,device,output):
    infos=run_episode(policy,env,device)
    write_json(output,{'metrics':metrics_from_infos(infos),'daily':infos})
    return metrics_from_infos(infos)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True)
    p.add_argument('--data',default='../data/nifty50_prices.csv')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--steps',type=int,default=512)
    p.add_argument('--batch-size',type=int,default=4)
    p.add_argument('--no-lora-train',action='store_true')
    p.add_argument('--lora-only',action='store_true',help='Causal ablation: freeze numeric encoders/readout, train only finance LoRA')
    p.add_argument('--lora-scope',choices=['attention','attention_mlp'],default='attention')
    args=p.parse_args()
    output=Path(args.output);output.mkdir(parents=True,exist_ok=False)
    seed_everything(args.seed);torch.set_num_threads(2)
    reward=FinanceRewardConfig(rebalance_rate=1.,rebalance_interval=21)
    env=PortfolioEnv(args.data,end_date='2022-12-31',covariance_window=252,reward_config=reward)
    env.reset()
    train,holdout,_=label_indices(len(env.prices),252,0)
    # Fixed holdout dates, chosen solely within pre-2023 history.
    probe=training_examples(env,holdout[np.linspace(0,len(holdout)-1,32,dtype=int)])
    validation=PortfolioEnv(args.data,start_date='2023-01-01',end_date='2024-12-31',
                           covariance_window=252,reward_config=reward)
    if args.lora_only and args.no_lora_train:raise ValueError('Cannot disable every trainable component')
    cfg=ModelConfig(dtype='fp32',actor_init_scale=1. if args.lora_only else .01,score_scale=1.,
                    use_asset_ids=False,normalize_market=True,covariance_window=252,
                    lora_only=args.lora_only,state_invariant=args.lora_only,lora_scope=args.lora_scope)
    policy=InternVLFinancePolicy(cfg,env.tickers,device='cuda')
    with torch.no_grad():
        policy.critic.weight.zero_();policy.critic.bias.zero_()
    policy.critic.requires_grad_(False);policy.log_std.requires_grad_(False)
    lora=[p for n,p in policy.shared.named_parameters() if p.requires_grad and 'lora_' in n]
    if args.no_lora_train:
        for parameter in lora:parameter.requires_grad_(False)
    optimizer=torch.optim.AdamW([
        {'params':[] if args.lora_only else policy.actor.parameters(),'lr':0. if args.lora_only else 5e-4,'weight_decay':0.},
        {'params':[] if args.lora_only else policy.encoders.parameters(),'lr':0. if args.lora_only else 1e-4,'weight_decay':1e-4},
        {'params':[] if args.no_lora_train else lora,'lr':2e-4 if args.lora_only else 2e-5,'weight_decay':0.}])
    metadata={'algorithm':'causal_minimum_variance_distillation_not_PPO','seed':args.seed,
        'train_end':'2022-12-31','covariance_window':252,'execution_interval':21,'batch_size':args.batch_size,
        'source_sha256':{f.name:digest(f) for f in Path(__file__).parent.glob('*.py')},
        'dataset_sha256':digest(args.data),'initial_lora_sha256':lora_digest(policy),
        'selection':'lowest pre-2023 holdout teacher KL; earlier checkpoint on tie',
        'no_lora_train':args.no_lora_train,'validation_role':'reused development, not final test',
        'lora_only':args.lora_only,
        'lora_scope':args.lora_scope,
        'initial_actor_scale':cfg.actor_init_scale,
        'teacher':'LedoitWolf covariance of 252 past returns, constrained long-only minimum variance, cap .30',
        'limitation':'fixed ticker universe; ordered covariance columns, no permutation-invariance claim',
        'optimizer':{'actor_lr':5e-4,'encoder_lr':1e-4,'lora_lr':2e-5,'gradient_clip':1.}}
    if args.lora_only:
        metadata['optimizer'].update(actor_lr=0.,encoder_lr=0.,lora_lr=2e-4)
        metadata['adaptation_scope']='Finance LoRA only; numeric encoders, actor, critic and exploration are frozen'
    import importlib.metadata
    metadata['package_versions']={name:importlib.metadata.version(name) for name in
        ('torch','transformers','peft','numpy','pandas','scikit-learn','scipy')}
    def numeric_digest():
        checksum=hashlib.sha256()
        for name,value in sorted(policy.named_parameters()):
            if not name.startswith('shared.'):
                checksum.update(name.encode());checksum.update(memoryview(value.detach().contiguous().view(torch.uint8).cpu().numpy()))
        return checksum.hexdigest()
    if args.lora_only:
        metadata['initial_frozen_numeric_sha256']=numeric_digest()
        if any('lora_' not in name for name,p in policy.named_parameters() if p.requires_grad):
            raise RuntimeError('LoRA-only experiment contains another trainable module')
    write_json(output/'run.json',metadata)
    initial={'fidelity':fidelity(policy,probe,'cuda'),
        'metrics':evaluate(policy,validation,'cuda',output/'initial_daily.json')}
    write_json(output/'initial.json',initial)
    policy.save_artifacts(output/'initial',{'algorithm':'supervised_initialization'},asdict(reward),
        {**metadata,'optimizer_updates':0,'learning_verified':False})
    rng=np.random.default_rng(args.seed)
    logs=[];candidates=[];started=time.monotonic()
    for step in range(1,args.steps+1):
        wait_for_compute_window()
        obs,target=training_examples(env,rng.choice(train,args.batch_size))
        scores,_=policy({k:v.to('cuda') for k,v in obs.items()})
        loss=-(target.to('cuda')*torch.log_softmax(scores,-1)).sum(-1).mean()
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite teacher loss')
        optimizer.zero_grad(set_to_none=True);loss.backward()
        norm=torch.nn.utils.clip_grad_norm_([p for p in policy.parameters() if p.requires_grad],1.,error_if_nonfinite=True)
        optimizer.step()
        if step==1 or step%32==0:
            row={'step':step,'cross_entropy':float(loss.detach()),'grad_norm':float(norm),'seconds':time.monotonic()-started,
                 'peak_allocated_GiB':torch.cuda.max_memory_allocated()/2**30}
            logs.append(row);write_json(output/'losses.json',logs);print(json.dumps(row),flush=True)
        if step%128==0 or step==args.steps:
            checkpoint=output/f'checkpoint_{step}'
            current=lora_digest(policy)
            run={**metadata,'steps':step,'optimizer_updates':step,'final_lora_sha256':current,
                 'learning_verified':current!=metadata['initial_lora_sha256']}
            if args.lora_only:
                run['final_frozen_numeric_sha256']=numeric_digest()
                if run['final_frozen_numeric_sha256']!=metadata['initial_frozen_numeric_sha256']:
                    raise RuntimeError('A frozen numeric encoder/readout changed')
            policy.save_artifacts(checkpoint,{'algorithm':'causal_teacher_distillation'},asdict(reward),run)
            diagnostic=fidelity(policy,probe,'cuda')
            metrics=evaluate(policy,validation,'cuda',output/f'development_{step}.json')
            row={'step':step,'package':str(checkpoint),'fidelity':diagnostic,'metrics':metrics}
            candidates.append(row);write_json(output/'candidates.json',candidates)
            print('CHECKPOINT',json.dumps(row),flush=True)
    selected=min(candidates,key=lambda row:(row['fidelity']['teacher_KL'],row['step']))
    write_json(output/'selection.json',{'criterion':metadata['selection'],'selected':selected,'initial':initial})
    write_json(output/'status.json',{'status':'completed','role':'development_supervised_experiment',
        'financial_generalization_proven':False,'test2025_accessed':False})


if __name__=='__main__':main()
