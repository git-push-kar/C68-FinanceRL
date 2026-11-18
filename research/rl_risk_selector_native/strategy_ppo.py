"""Weekly strategy-selection PPO. New exploratory branch; no fresh-test claim."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from env import PortfolioEnv,FinanceRewardConfig
from model import InternVLFinancePolicy,write_json,digest
from config import ModelConfig
from train import seed_everything,lora_digest
from metrics import metrics_from_infos
from execution_policy import wait_for_compute_window,configure_finance_precision
from ppo import compute_gae
from allocation import allocation_weights

NAMES=['equal_weight','inverse_volatility','momentum_126_skip21','reversal_5']

class ExecutionEnv(PortfolioEnv):
    hold_only=False
    def _target_weights(self,action):
        if self.hold_only:return self.weights.copy()
        return super()._target_weights(action)

class Market:
    def __init__(self,data,start,end):
        self.env=ExecutionEnv(data,start_date=start,end_date=end,window=30,
            reward_config=FinanceRewardConfig(turnover_penalty=0,volatility_penalty=0,drawdown_penalty=0))
        if self.env.start_index<253:raise ValueError('Need 253 prior closes')
        self.cache={}
    def reset(self,index=None):
        self.env.reset()
        if index is not None:self.env.index=int(index)
        self.infos=[]
        return self.observe()
    def observe(self):
        e=self.env;i=e.index-1
        if i not in self.cache:
            e.index=i;obs=e._observation();e.index=i+1
            p=e.prices;returns=p[i-125:i+1]/p[i-126:i]-1
            vol=returns.std(0).clip(1e-5)
            mom=p[i-21]/p[i-126]-1;rev=-(p[i]/p[i-5]-1)
            def tilt(x):
                z=(x-x.mean())/(x.std()+1e-8);w=np.exp(np.clip(z,-2,2));return w/w.sum()
            inv=1/vol;inv/=inv.sum()
            targets=np.stack([np.ones(e.n_assets)/e.n_assets,inv,tilt(mom),tilt(rev)])
            # Soft bounded tilts mixed with diversification keep all targets under .30.
            targets=.5*targets+.5/e.n_assets
            targets=np.stack([allocation_weights(np.log(w),.30) for w in targets])
            assert targets.max()<=.30 and np.allclose(targets.sum(1),1)
            market=obs['market'].copy()
            # Explicit long/short signals replace redundant return/MA features.
            market[:,:,1]=np.broadcast_to(mom,market[:,:,1].shape)
            market[:,:,3]=np.broadcast_to(rev,market[:,:,3].shape)
            self.cache[i]=(market,targets)
        market,targets=self.cache[i]
        # Current holdings are known before executing; market signal is one close old.
        obs={'market':market.copy(),'portfolio':np.r_[e.weights,e.cash_weight].astype('float32'),
             'risk':np.array([np.std(e.returns[-20:]) if len(e.returns)>1 else 0,
                 1-e.equity/e.peak_equity,e.equity/e.initial_cash-1],dtype='float32')}
        obs['candidate_targets']=targets.astype('float32')
        return obs,targets
    def step(self,target):
        e=self.env;reward=0.;done=False
        for day in range(5):
            e.hold_only=day!=0
            action=np.log(np.maximum(target if day==0 else e.weights,1e-30))
            _,_,done,_,info=e.step(action)
            reward+=100*(np.log1p(info['net_return'])-np.log1p(info['benchmark_net_return']))
            info['signal_close_date']=str(e.dates[e.index-2-day])[:10]
            self.infos.append(info)
            if done:break
        return (None if done else self.observe()),reward,done

class Policy(nn.Module):
    def __init__(self,tickers,numeric=False):
        super().__init__();self.numeric=numeric
        if numeric:
            self.features=nn.Sequential(nn.Linear(30*len(tickers)*5+len(tickers)+4,128),nn.Tanh(),nn.Linear(128,128),nn.Tanh())
            hidden=128
        else:
            self.base=InternVLFinancePolicy(ModelConfig(dtype='fp32',use_asset_ids=False,normalize_market=True),tickers,device='cuda')
            self.base.actor.requires_grad_(False);self.base.critic.requires_grad_(False);self.base.log_std.requires_grad_(False)
            hidden=self.base.shared.hidden_size
        self.gate=nn.Linear(hidden,4);self.value=nn.Linear(hidden,1)
        nn.init.zeros_(self.gate.weight);nn.init.zeros_(self.gate.bias)
        nn.init.zeros_(self.value.weight);nn.init.zeros_(self.value.bias)
        device='cpu' if numeric else 'cuda'
        self.gate.to(device);self.value.to(device)
    def forward(self,obs):
        if self.numeric:
            h=self.features(torch.cat([obs['market'].flatten(1),obs['portfolio'],obs['risk']],1))
        else:
            tokens=self.base.encoders(obs['market'],obs['portfolio'],obs['risk'])
            h=self.base.shared.finance_hidden_states(tokens)[:,-1].float()
        return torch.distributions.Categorical(logits=self.gate(h)),self.value(h.detach()).squeeze(-1)

def tensor(obs,device):return {k:torch.as_tensor(v,device=device).unsqueeze(0) for k,v in obs.items()}

@torch.no_grad()
def evaluate(policy,market,device,fixed=None):
    state=market.reset();probs=[]
    while True:
        wait_for_compute_window();obs,targets=state
        p=np.eye(4)[fixed] if isinstance(fixed,int) else np.ones(4)/4 if fixed=='uniform' else policy(tensor(obs,device))[0].probs[0].cpu().numpy()
        # Stochastic policy's expected target; same declared rule before and after training.
        probs.append(p.tolist());state,_,done=market.step(p@targets)
        if done:break
    return {'metrics':metrics_from_infos(market.infos),'daily':market.infos,'probabilities':probs}

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--numeric',action='store_true')
    p.add_argument('--seed',type=int,default=42);p.add_argument('--iterations',type=int,default=16)
    p.add_argument('--eval-every',type=int,default=4)
    p.add_argument('--no-lora-train',action='store_true')
    p.add_argument('--data',default='../sector_data_20261107/sector9_development_through2024.csv');a=p.parse_args()
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False);seed_everything(a.seed);configure_finance_precision();torch.set_num_threads(2)
    train=Market(a.data,'2001-01-01','2018-12-31');val=Market(a.data,'2019-01-01','2021-12-31')
    dev=Market(a.data,'2022-01-01','2024-12-31');device='cpu' if a.numeric else 'cuda'
    policy=Policy(train.env.tickers,a.numeric)
    if a.no_lora_train and not a.numeric:
        for name,param in policy.base.shared.named_parameters():
            if 'lora_' in name:param.requires_grad_(False)
    params=[x for n,x in policy.named_parameters() if x.requires_grad and not n.startswith('value.')]
    optimizer=torch.optim.AdamW(params,lr=1e-4 if a.numeric else 1e-5,weight_decay=0)
    value_optimizer=torch.optim.Adam(policy.value.parameters(),lr=1e-3)
    meta={'algorithm':'categorical_strategy_selection_PPO','seed':a.seed,'numeric':a.numeric,'strategies':NAMES,'arguments':vars(a),
        'training':'2001-2018','selection':'2019-2021 Sharpe; unchanged initial eligible','development':'2022-2024 exploratory',
        'final_test':'NONE; May2025-Jan2026 already used, never accessed by this driver',
        'signal_lag_closes':1,'decision_interval':5,'reward':'100 times net log return minus daily equal-weight net log return',
        'critic_backbone_gradients':False,'evaluation':'expected strategy mixture, all policies including initialization',
        'dataset_sha256':digest(a.data),'source_sha256':{f.name:digest(f) for f in Path(__file__).parent.glob('*.py')},
        'initial_lora_sha256':None if a.numeric else lora_digest(policy.base)}
    write_json(out/'protocol.json',meta)
    for name,fixed in [('uniform', 'uniform')]+list(zip(NAMES,range(4))):
        write_json(out/f'baseline_{name}.json',evaluate(policy,dev,device,fixed))
    candidates=[];updates=0;rng=np.random.default_rng(a.seed);started=time.monotonic()
    def assess(step):
        result=evaluate(policy,val,device);development=evaluate(policy,dev,device)
        write_json(out/f'validation_{step}.json',result);write_json(out/f'development_{step}.json',development)
        package=out/f'checkpoint_{step}';package.mkdir()
        if not a.numeric:
            policy.base.save_artifacts(package/'base_modules',{}, {},meta)
            torch.save({'gate':policy.gate.state_dict(),'value':policy.value.state_dict()},package/'strategy_heads.pt')
        else:torch.save(policy.state_dict(),package/'numeric.pt')
        row={'step':step,'validation':result['metrics'],'development':development['metrics'],'package':str(package.resolve())}
        candidates.append(row);write_json(out/'candidates.json',candidates);print('ASSESS',json.dumps(row),flush=True)
    assess(0)
    for iteration in range(1,a.iterations+1):
        observations=[];actions=[];logps=[];values=[];rewards=[];dones=[]
        start=int(rng.integers(train.env.start_index,len(train.env.prices)-161));state=train.reset(start)
        for t in range(32):
            wait_for_compute_window();obs,targets=state
            with torch.no_grad():dist,value=policy(tensor(obs,device));action=dist.sample()
            observations.append(obs);actions.append(action.item());logps.append(dist.log_prob(action).item());values.append(value.item())
            state,reward,done=train.step(targets[action.item()]);rewards.append(reward);dones.append(float(done))
            if done:break
        with torch.no_grad():bootstrap=0. if done else policy(tensor(state[0],device))[1].item()
        adv,returns=compute_gae(torch.tensor(rewards),torch.tensor(values+[bootstrap]),torch.tensor(dones),.99,.95)
        adv=(adv-adv.mean())/(adv.std(unbiased=False)+1e-8);initial_error=0.
        # Match single-sample kernels and accumulate one true minibatch per optimizer update.
        for epoch in range(2):
            optimizer.zero_grad();value_optimizer.zero_grad();kls=[]
            for i in rng.permutation(len(actions)):
                wait_for_compute_window();dist,value=policy(tensor(observations[i],device))
                logp=dist.log_prob(torch.tensor([actions[i]],device=device));delta=logp-logps[i]
                if epoch==0:initial_error=max(initial_error,abs(delta.item()))
                ratio=delta.exp();surrogate=torch.minimum(ratio*adv[i].to(device),ratio.clamp(.8,1.2)*adv[i].to(device))
                loss=(-surrogate-.005*dist.entropy()+.5*(value-returns[i].to(device)).square()).mean()/len(actions)
                loss.backward();kls.append(((ratio-1)-delta).item())
            if epoch==0 and initial_error>1e-4:raise RuntimeError(f'Initial likelihood mismatch {initial_error}')
            if np.mean(kls)>.03:break
            torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True)
            torch.nn.utils.clip_grad_norm_(policy.value.parameters(),1.,error_if_nonfinite=True)
            optimizer.step();value_optimizer.step();updates+=1
        row={'iteration':iteration,'updates':updates,'reward_mean':float(np.mean(rewards)),'initial_logp_error':initial_error,
             'mean_KL':float(np.mean(kls)),'seconds':time.monotonic()-started}
        write_json(out/'progress.json',row);print(json.dumps(row),flush=True)
        if iteration%a.eval_every==0 or iteration==a.iterations:assess(iteration)
    selected=max(candidates,key=lambda x:(x['validation']['sharpe_proxy'],-x['step']))
    write_json(out/'selection.json',{'selected':selected,'initial':candidates[0],'criterion':'2019-2021 Sharpe only; no final test claim'})
    write_json(out/'status.json',{'status':'completed','optimizer_updates':updates,'initial_lora_sha256':meta['initial_lora_sha256'],
        'final_lora_sha256':None if a.numeric else lora_digest(policy.base),'financial_goal_met':False})

if __name__=='__main__':main()
