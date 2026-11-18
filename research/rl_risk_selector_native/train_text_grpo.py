"""Critic-free grouped trajectory RL for causal stock-sector allocation.

Independent implementation inspired by group-relative policy optimization.
Each group shares a historical start but samples different sequential actions.
Future prices are rewards only; holdings/costs evolve along each trajectory.
"""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from strategy_ppo import Policy,Market,evaluate,tensor,NAMES
from train import seed_everything,lora_digest
from model import write_json,digest
from execution_policy import wait_for_compute_window,configure_finance_precision
from text_policy import TextPolicy

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--numeric',action='store_true')
    p.add_argument('--no-lora-train',action='store_true');p.add_argument('--seed',type=int,default=42)
    p.add_argument('--updates',type=int,default=64);p.add_argument('--eval-every',type=int,default=16)
    p.add_argument('--group-size',type=int,default=4);p.add_argument('--decisions',type=int,default=12)
    a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    seed_everything(a.seed);configure_finance_precision();torch.set_num_threads(2)
    data='../sector_data_20261107/sector9_development_through2024.csv'
    train=Market(data,'2001-01-01','2018-12-31');val=Market(data,'2019-01-01','2021-12-31')
    dev=Market(data,'2022-01-01','2024-12-31');device='cpu' if a.numeric else 'cuda'
    if a.numeric or a.no_lora_train:raise ValueError('This arm trains only native-text LoRA; use explicit numeric controls separately')
    policy=TextPolicy(train.env.tickers)
    groups=[{'params':[p for p in policy.parameters() if p.requires_grad],'lr':1e-5}]
    optimizer=torch.optim.AdamW(groups,weight_decay=0)
    rng=np.random.default_rng(a.seed);candidates=[];optimizer_updates=0
    initial_lora=lora_digest(policy)
    meta={'algorithm':'clipped_group_relative_sequential_trajectory_policy_gradient','arguments':vars(a),
          'train':'2001-2018','selection':'2019-2021 net Sharpe, unchanged initial eligible','development':'2022-2024 exploratory',
          'reward':f'Annualized Sharpe of {a.decisions*5} daily net returns; zero risk-free rate',
          'policy':'Native InternVL text embeddings and frozen selected original vocabulary rows; only attention LoRA trains',
          'group_advantage':'Centered by same-start group mean, divided by max(group std,.05); no critic',
          'cost':'10bp times full traded-stock L1, holdings drift, one-close delayed market signals',
          'actions':NAMES,'evaluation':'Expected strategy mixture; stochastic robustness required before promotion',
          'source_sha256':{f.name:digest(f) for f in Path(__file__).parent.glob('*.py')},
          'dataset_sha256':digest(data),'test2026_accessed':False,'initial_lora_sha256':initial_lora,
          'prior_art':'https://arxiv.org/abs/2608.09880 ; GRPO is established, our multi-decision trajectory variant is not claimed novel without further comparison'}
    write_json(out/'protocol.json',meta)
    def assess(step):
        v=evaluate(policy,val,device);d=evaluate(policy,dev,device)
        write_json(out/f'validation_{step}.json',v);write_json(out/f'development_{step}.json',d)
        package=out/f'checkpoint_{step}';package.mkdir()
        policy.save(package,meta)
        torch.save({'optimizer':optimizer.state_dict(),'update':step,'numpy_generator':rng.bit_generator.state,
            'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all() if not a.numeric else []},package/'training_state.pt')
        row={'step':step,'validation':v['metrics'],'development':d['metrics'],'package':str(package.resolve())}
        candidates.append(row);write_json(out/'candidates.json',candidates);print('ASSESS',json.dumps(row),flush=True)
    assess(0);started=time.monotonic()
    for update in range(1,a.updates+1):
        start=int(rng.integers(train.env.start_index,len(train.env.prices)-a.decisions*5-1))
        trajectories=[];scores=[]
        for group in range(a.group_size):
            state=train.reset(start);trajectory=[]
            for _ in range(a.decisions):
                wait_for_compute_window();obs,targets=state
                with torch.no_grad():dist,_=policy(tensor(obs,device));action=dist.sample();logp=dist.log_prob(action)
                trajectory.append((obs,action.item(),logp.item()))
                state,_,done=train.step(targets[action.item()])
                if done:raise RuntimeError('Reward horizon crossed training boundary')
            returns=np.array([i['net_return'] for i in train.infos]);score=np.sqrt(252)*returns.mean()/max(returns.std(),1e-6)
            trajectories.append(trajectory);scores.append(float(score))
        advantages=(np.array(scores)-np.mean(scores))/max(np.std(scores),.05)
        initial_error=0.;kl=0.
        for epoch in range(2):
            optimizer.zero_grad();kls=[]
            for group,trajectory in enumerate(trajectories):
                for obs,action,oldlogp in trajectory:
                    wait_for_compute_window();dist,_=policy(tensor(obs,device))
                    delta=dist.log_prob(torch.tensor([action],device=device))-oldlogp
                    if epoch==0:initial_error=max(initial_error,abs(delta.item()))
                    ratio=delta.exp();adv=float(advantages[group])
                    loss=(-torch.minimum(ratio*adv,ratio.clamp(.8,1.2)*adv)-.005*dist.entropy()).mean()/(a.group_size*a.decisions)
                    if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
                    loss.backward();kls.append(((ratio-1)-delta).item())
            if epoch==0 and initial_error>1e-4:raise RuntimeError('Rollout likelihood mismatch')
            kl=float(np.mean(kls))
            if kl>.03:break
            torch.nn.utils.clip_grad_norm_([p for p in policy.parameters() if p.requires_grad],1.,error_if_nonfinite=True)
            optimizer.step();optimizer_updates+=1
        row={'update':update,'optimizer_updates':optimizer_updates,'group_mean_Sharpe':float(np.mean(scores)),
             'group_reward_std':float(np.std(scores)),'initial_logp_error':initial_error,'KL':kl,'seconds':time.monotonic()-started}
        write_json(out/'progress.json',row);print(json.dumps(row),flush=True)
        if update%a.eval_every==0 or update==a.updates:assess(update)
    selected=max(candidates,key=lambda x:(x['validation']['sharpe_proxy'],-x['step']))
    write_json(out/'selection.json',{'selected':selected,'initial':candidates[0],'criterion':meta['selection']})
    write_json(out/'status.json',{'status':'completed','optimizer_updates':optimizer_updates,
        'initial_lora_sha256':initial_lora,'final_lora_sha256':lora_digest(policy),
        'financial_goal_met':False,'test2026_accessed':False})

if __name__=='__main__':main()
