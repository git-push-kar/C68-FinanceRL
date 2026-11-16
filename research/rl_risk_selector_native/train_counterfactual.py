"""Finance RL with enumerated first actions and truncated Monte Carlo values.

Only native InternVL attention/readout LoRA learns. This is an independently
implemented all-action policy-gradient experiment, not a novelty claim.
"""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from counterfactual import action_values,choose
from strategy_ppo import Market,evaluate,tensor,NAMES
from text_policy import TextPolicy
from train import seed_everything,lora_digest
from model import write_json,digest
from execution_policy import configure_finance_precision,wait_for_compute_window

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--seed',type=int,default=42)
    p.add_argument('--updates',type=int,default=64);p.add_argument('--eval-every',type=int,default=16)
    p.add_argument('--segments',type=int,default=4);p.add_argument('--states-per-segment',type=int,default=4)
    p.add_argument('--burn-in',type=int,default=4);p.add_argument('--horizon',type=int,default=3)
    a=p.parse_args()
    if min(a.updates,a.eval_every,a.segments,a.states_per_segment,a.horizon)<1 or a.burn_in<0:
        raise ValueError('Positive budgets and nonnegative burn-in required')
    batch_states=a.segments*a.states_per_segment
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    seed_everything(a.seed);configure_finance_precision();torch.set_num_threads(2)
    data='../sector_data_20261107/sector9_development_through2024.csv'
    train=Market(data,'2001-01-01','2018-12-31');val=Market(data,'2019-01-01','2021-12-31')
    dev=Market(data,'2022-01-01','2024-12-31');policy=TextPolicy(train.env.tickers)
    params=[p for p in policy.parameters() if p.requires_grad]
    if any('lora_' not in n for n,p in policy.named_parameters() if p.requires_grad):raise AssertionError('Only LoRA may learn')
    head=[p for n,p in policy.named_parameters() if p.requires_grad and '.lm_head.' in n]
    attention=[p for n,p in policy.named_parameters() if p.requires_grad and '.lm_head.' not in n]
    if not head or not attention:raise AssertionError('Both declared adapter groups are required')
    optimizer=torch.optim.AdamW([{'params':attention,'lr':1e-5},{'params':head,'lr':3e-4}],weight_decay=0)
    rng=np.random.default_rng(a.seed);candidates=[];initial=lora_digest(policy)
    meta={'algorithm':'all_action_truncated_return_policy_gradient','arguments':vars(a),
        'training':'2001-2018','selection':'2019-2021 net Sharpe; unchanged initialization eligible',
        'development':'2022-2024 reused exploratory','policy':'Native original embeddings and vocabulary rows; attention plus original vocabulary-projection LoRA learns',
        'adapter_scope':'attention_readout_v1',
        'reward':'100 * net log wealth excess over daily equal-weight, same as earlier PPO reward',
        'action_values':f'Enumerate all four first actions, then {a.horizon-1} sampled continuation decisions under the current policy; gamma=.99; no bootstrap',
        'variance_control':'Common continuation uniforms across branches; center four action values; one common batch advantage standard deviation with .05 floor',
        'state_sampling':'Four independently started historical segments per update by default, four collected states each, with four earlier policy decisions as burn-in to establish realistic holdings',
        'optimization':'One fresh rollout batch per AdamW update; attention lr=1e-5, readout LoRA lr=3e-4; entropy coefficient=.005, gradient norm cap=1; fail if post-update exact categorical KL exceeds .03',
        'limitations':'Finite-horizon Monte Carlo values have truncation bias. No claim of exact full-horizon gradient or novel all-action estimator. Reward, horizon and budget differ from the GRPO feasibility pilot.',
        'prior_art':'https://jmlr.org/papers/v21/18-012.html',
        'cost':'10bp * full traded-stock L1; five-session decisions; drift; one-close market-signal delay',
        'actions':NAMES,'evaluation':'Same expected-strategy-mixture rule before and after RL',
        'initial_lora_sha256':initial,'dataset_sha256':digest(data),
        'source_sha256':{f.name:digest(f) for f in Path(__file__).parent.glob('*.py')},'test2026_accessed':False}
    write_json(out/'protocol.json',meta)
    @torch.no_grad()
    def probabilities(obs):
        wait_for_compute_window();return policy(tensor(obs,'cuda'))[0].probs[0].cpu().numpy()
    def assess(step):
        v=evaluate(policy,val,'cuda');d=evaluate(policy,dev,'cuda')
        write_json(out/f'validation_{step}.json',v);write_json(out/f'development_{step}.json',d)
        package=out/f'checkpoint_{step}';policy.save(package,meta)
        torch.save({'optimizer':optimizer.state_dict(),'update':step,'numpy_generator':rng.bit_generator.state,
            'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all()},package/'training_state.pt')
        row={'step':step,'validation':v['metrics'],'development':d['metrics'],'package':str(package.resolve())}
        candidates.append(row);write_json(out/'candidates.json',candidates);print('ASSESS',json.dumps(row),flush=True)
    assess(0);started=time.monotonic()
    for update in range(1,a.updates+1):
        observations=[];old=[];values=[]
        for segment in range(a.segments):
            start=int(rng.integers(train.env.start_index,len(train.env.prices)-(a.burn_in+a.states_per_segment+a.horizon)*5-1))
            state=train.reset(start)
            for _ in range(a.burn_in):
                obs,targets=state;prob=probabilities(obs)
                state,_,done=train.step(targets[choose(prob,rng.random())])
                if done:raise RuntimeError('Burn-in crossed training boundary')
            for _ in range(a.states_per_segment):
                obs,targets=state;prob=probabilities(obs)
                values.append(action_values(train,targets,probabilities,rng.random(a.horizon-1)))
                observations.append(obs);old.append(prob.copy())
                state,_,done=train.step(targets[choose(prob,rng.random())])
                if done:raise RuntimeError('Actual rollout crossed training boundary')
        q=np.stack(values);centered=q-q.mean(1,keepdims=True);scale=max(float(centered.std()),.05)
        advantage=centered/scale;optimizer.zero_grad();error=0.
        for obs,prior,adv in zip(observations,old,advantage):
            wait_for_compute_window();dist,_=policy(tensor(obs,'cuda'))
            error=max(error,float(np.max(abs(dist.probs[0].detach().cpu().numpy()-prior))))
            loss=(-(dist.probs[0]*torch.as_tensor(adv,device='cuda')).sum()-.005*dist.entropy().mean())/batch_states
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite policy loss')
            loss.backward()
        if error>1e-7:raise AssertionError('Collection/update probability mismatch')
        grad=float(torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True));optimizer.step()
        kls=[];post_probabilities=[]
        for obs,prior in zip(observations,old):
            post=probabilities(obs).astype(float);prior=prior.astype(float)
            post_probabilities.append(post)
            kls.append(float(np.sum(prior*(np.log(prior)-np.log(post)))))
        kl=float(np.mean(kls))
        row={'update':update,'optimizer_updates':update,'initial_probability_error':error,'mean_post_update_KL':kl,
            'gradient_norm':grad,'advantage_scale':scale,'mean_action_value_spread':float(np.mean(q.max(1)-q.min(1))),
            'mean_centered_action_values':centered.mean(0).tolist(),
            'mean_policy_before':np.asarray(old).mean(0).tolist(),
            'mean_policy_after':np.asarray(post_probabilities).mean(0).tolist(),
            'seconds':time.monotonic()-started}
        write_json(out/'progress.json',row);print(json.dumps(row),flush=True)
        if kl>.03:raise RuntimeError('Update exceeded declared KL guard; run failed, do not promote')
        if update%a.eval_every==0 or update==a.updates:assess(update)
    selected=max(candidates,key=lambda c:(c['validation']['sharpe_proxy'],-c['step']))
    write_json(out/'selection.json',{'selected':selected,'initial':candidates[0],'criterion':meta['selection']})
    write_json(out/'status.json',{'status':'completed','optimizer_updates':a.updates,
        'initial_lora_sha256':initial,'final_lora_sha256':lora_digest(policy),
        'financial_goal_met':False,'test2026_accessed':False})

if __name__=='__main__':main()
