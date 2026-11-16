"""Direct, anchor-regularized portfolio policy improvement, not PPO/GRPO."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time
import numpy as np
import torch

from classical_panel import min_variance
from differentiable_portfolio import capped_weights,portfolio_step
from env import PortfolioEnv,FinanceRewardConfig
from execution_policy import wait_for_compute_window
from model import InternVLFinancePolicy,digest,write_json
from train import seed_everything,lora_digest
from train_covariance_teacher import training_examples,fidelity,evaluate
from warmup_finance import label_indices


def rollout_loss(policy,env,start,horizon,anchor_coefficient=5.,device='cuda',signal_lag=0):
    """All future prices are training rewards, never observation features.

    Holdings, risk statistics, equity and future rebalance costs retain their
    complete autograd path; no hidden truncation of the policy state gradient.
    """
    env.reset()
    holdings=torch.zeros(env.n_assets,dtype=torch.float64,device=device)
    cash=holdings.new_tensor(1.)
    equity=holdings.new_tensor(1.);peak=equity
    if signal_lag not in (0,1):raise ValueError('Expected zero or one signal-close delay')
    nets=[];anchors=[];previous_state=(holdings,cash,equity,peak,[])
    for day in range(horizon):
        index=start+day
        current_state=(holdings,cash,equity,peak,nets[-20:])
        if day%21==0:
            env.index=index-signal_lag
            static=env._observation()
            signal_holdings,signal_cash,signal_equity,signal_peak,signal_nets=previous_state if signal_lag else current_state
            volatility=torch.stack(signal_nets).std(unbiased=False) if len(signal_nets)>1 else equity.new_tensor(0.)
            risk=torch.stack((volatility,1-signal_equity/signal_peak,signal_equity-1))
            obs={'market':torch.as_tensor(static['market'],device=device).unsqueeze(0),
                 'covariance':torch.as_tensor(static['covariance'],device=device).unsqueeze(0),
                 'portfolio':torch.cat((signal_holdings,signal_cash.view(1))).float().unsqueeze(0),
                 'risk':risk.float().unsqueeze(0)}
            scores,_=policy(obs)
            target=capped_weights(scores[0],env.reward_config.max_weight)
            teacher=torch.as_tensor(min_variance(static['covariance'].astype(np.float64),
                cap=env.reward_config.max_weight),device=device)
            anchors.append((teacher*(teacher.clamp_min(1e-12).log()-target.clamp_min(1e-12).log())).sum())
        else:target=holdings
        previous_state=current_state
        relatives=torch.as_tensor(env.prices[index+1]/env.prices[index],device=device)
        net,holdings,_=portfolio_step(target,holdings,relatives,env.reward_config.transaction_cost_bps)
        cash=cash*0
        nets.append(net);equity=equity*(1+net);peak=torch.maximum(peak,equity)
    returns=torch.stack(nets)
    sharpe=np.sqrt(252)*returns.mean()/returns.std(unbiased=False).clamp_min(1e-6)
    anchor=torch.stack(anchors).mean()
    loss=-sharpe+anchor_coefficient*anchor
    return loss,{'sharpe':float(sharpe.detach()),'anchor_KL':float(anchor.detach()),
                 'return':float((equity-1).detach())}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent-run',required=True);p.add_argument('--output',required=True)
    p.add_argument('--data',default='../data/nifty50_prices.csv');p.add_argument('--seed',type=int,default=42)
    p.add_argument('--updates',type=int,default=128);p.add_argument('--horizon',type=int,default=126)
    p.add_argument('--windows-per-update',type=int,default=4)
    p.add_argument('--anchor-coefficient',type=float,default=5.)
    args=p.parse_args()
    output=Path(args.output);output.mkdir(parents=True,exist_ok=False)
    selection=json.loads((Path(args.parent_run)/'selection.json').read_text())
    parent=Path(selection['selected']['package']).resolve()
    seed_everything(args.seed);torch.set_num_threads(2)
    cfg=json.loads((parent/'environment_config.json').read_text())
    reward=FinanceRewardConfig(**cfg['reward'])
    env=PortfolioEnv(args.data,end_date='2022-12-31',covariance_window=252,reward_config=reward);env.reset()
    train,holdout,boundary=label_indices(len(env.prices),252,0)
    starts=np.arange(252,boundary-args.horizon)
    if len(starts)<2:raise ValueError('Training windows do not fit before chronological holdout')
    assert starts[-1]+args.horizon<boundary
    holdout_start=str(env.dates[boundary])[:10]
    internal=PortfolioEnv(args.data,start_date=holdout_start,end_date='2022-12-31',covariance_window=252,reward_config=reward)
    development=PortfolioEnv(args.data,start_date='2023-01-01',end_date='2024-12-31',covariance_window=252,reward_config=reward)
    probe=training_examples(env,holdout[np.linspace(0,len(holdout)-1,32,dtype=int)])
    policy=InternVLFinancePolicy.load_artifacts(parent,device='cuda',for_training=True)
    policy.critic.requires_grad_(False);policy.log_std.requires_grad_(False)
    lora=[p for n,p in policy.shared.named_parameters() if p.requires_grad and 'lora_' in n]
    optimizer=torch.optim.AdamW([
        {'params':policy.actor.parameters(),'lr':5e-5,'weight_decay':0.},
        {'params':policy.encoders.parameters(),'lr':1e-5,'weight_decay':1e-4},
        {'params':lora,'lr':2e-6,'weight_decay':0.}])
    metadata={'algorithm':'direct_differentiable_Sharpe_with_causal_teacher_anchor_not_PPO',
        'train_end':'2022-12-31','parent_manifest_sha256':digest(parent/'manifest.json'),'parent_package':str(parent),
        'seed':args.seed,'arguments':vars(args),'holdout_first_return':holdout_start,
        'last_training_reward_date':str(env.dates[starts[-1]+args.horizon])[:10],
        'selection':'pre-2023 holdout Sharpe minus anchor_coefficient times teacher KL; includes unchanged parent',
        'source_sha256':{f.name:digest(f) for f in Path(__file__).parent.glob('*.py')},
        'dataset_sha256':digest(args.data),'initial_lora_sha256':lora_digest(policy),
        'effective_batch':f'{args.windows_per_update} full training windows, one clip/optimizer step; exact gradient accumulation',
        'optimizer':{'actor_lr':5e-5,'encoder_lr':1e-5,'lora_lr':2e-6,'gradient_clip':1.},
        'critic_trained':False,'test2025_accessed':False}
    write_json(output/'run.json',metadata)
    candidates=[]
    def assess(step,package):
        teacher=fidelity(policy,probe,'cuda')
        hold=evaluate(policy,internal,'cuda',output/f'holdout_{step}.json')
        dev=evaluate(policy,development,'cuda',output/f'development_{step}.json')
        row={'step':step,'package':str(package),'fidelity':teacher,'internal_metrics':hold,'metrics':dev,
            'selection_utility':hold['sharpe_proxy']-args.anchor_coefficient*teacher['teacher_KL']}
        candidates.append(row);write_json(output/'candidates.json',candidates)
        print('ASSESSMENT',json.dumps(row),flush=True)
    assess(0,parent)
    rng=np.random.default_rng(args.seed)
    started=time.monotonic();logs=[]
    for update in range(1,args.updates+1):
        optimizer.zero_grad(set_to_none=True);samples=[]
        for start in rng.choice(starts,args.windows_per_update):
            wait_for_compute_window()
            loss,details=rollout_loss(policy,env,int(start),args.horizon,args.anchor_coefficient)
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite utility objective')
            (loss/args.windows_per_update).backward();samples.append(details)
            del loss
        norm=torch.nn.utils.clip_grad_norm_([p for p in policy.parameters() if p.requires_grad],1.,error_if_nonfinite=True)
        optimizer.step()
        if update==1 or update%8==0:
            row={'updates':update,'sample_mean_sharpe':float(np.mean([s['sharpe'] for s in samples])),
                'sample_mean_anchor_KL':float(np.mean([s['anchor_KL'] for s in samples])),
                'grad_norm':float(norm),'seconds':time.monotonic()-started,
                'peak_allocated_GiB':torch.cuda.max_memory_allocated()/2**30}
            logs.append(row);write_json(output/'losses.json',logs);print(json.dumps(row),flush=True)
        if update%32==0 or update==args.updates:
            package=output/f'checkpoint_{update}'
            current=lora_digest(policy)
            policy.save_artifacts(package,{'algorithm':'direct_policy_utility'},asdict(reward),
                {**metadata,'steps':update,'optimizer_updates':update,'final_lora_sha256':current,
                 'learning_verified':current!=metadata['initial_lora_sha256']})
            assess(update,package)
    selected=max(candidates,key=lambda row:(row['selection_utility'],-row['step']))
    write_json(output/'selection.json',{'selected':selected,'initial':candidates[0],
        'criterion':metadata['selection'],'all_candidates':candidates})
    write_json(output/'status.json',{'status':'completed','test2025_accessed':False,
        'refinement_selected':selected['step']>0})


if __name__=='__main__':main()
