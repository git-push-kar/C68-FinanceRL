"""Factorized clipped policy gradient from complete evolving49-stock accounts."""
import numpy as np
import torch
from market import LEVELS,HORIZON

def clipped_loss(p,old,ref,actions,advantage):
 a=torch.as_tensor(actions,dtype=torch.long,device=p.device)[:,None]
 ratio=torch.exp((p.gather(1,a).log()-old.gather(1,a).log()).clamp(-20,20))
 advantage=p.new_tensor(float(advantage));surrogate=torch.minimum(ratio*advantage,ratio.clamp(.8,1.2)*advantage).mean()
 kl=(p*(p.log()-ref.log())).sum(-1).mean();oldkl=(old*(old.log()-p.log())).sum(-1).mean();entropy=-(p*p.log()).sum(-1).mean()
 return -surrogate+.01*kl-.003*entropy,{'ratio':float(ratio.detach().mean()),'reference_kl':float(kl.detach()),'old_kl':float(oldkl.detach()),'entropy':float(entropy.detach())}

class WindowSampler:
 def __init__(self,m,seed):
  self.rng=np.random.default_rng(seed);self.buckets={};first,last=m.bounds('2018-07-02','2023-12-31')
  for t in range(first,last-60+1,5):
   trend=float(np.mean(m.prices[t]/m.prices[t-60]-1));daily=(m.prices[t-19:t+1]/m.prices[t-20:t]-1).mean(1);vol=float(daily.std()*np.sqrt(252))
   key=('up' if trend>=0 else 'down')+('_high' if vol>=.2 else '_low');self.buckets.setdefault(key,[]).append(t)
  self.keys=sorted(self.buckets)
 def choose(self,i):
  key=self.keys[i%len(self.keys)];return int(self.rng.choice(self.buckets[key])),float(self.rng.choice([5000,50000,100000,200000])),key

def rollout(policy,m,t,capital,rng):
 c=capital;q=np.zeros(49,int);pending=np.zeros(2);records=[];summary=[];values=[];begin=t
 for _ in range(3):
  obs=m.observe(t,c,q,capital,pending);e=policy.encode(obs,permutation=rng.permutation(49))
  with torch.no_grad():
   p=policy(e);actions=torch.distributions.Categorical(probs=p).sample().cpu().numpy();old=p.cpu().numpy()
   with policy.reference_context():ref=policy(e).cpu().numpy()
  r=m.horizon(t,c,q,LEVELS[actions],t+HORIZON,pending);records.append({'encoded':e,'old':old,'reference':ref,'actions':actions})
  summary.append({'signal_date':obs['asof'],'actions':actions.tolist(),'trade_shares':r['trade_shares'].tolist(),'prompt_sha256':e['prompt_sha256'],'tokens':e['ids'].shape[1]})
  c,q,pending=r['cash'],r['qty'],r['pending'];values.extend(r['values'].tolist());t+=HORIZON
 path=np.r_[capital,values];dd=float((1-path/np.maximum.accumulate(path)).max());growth=float(np.log(path[-1]/capital))
 return records,{'begin':str(m.dates[begin]),'end':str(m.dates[t]),'capital':capital,'return':float(path[-1]/capital-1),'drawdown':dd,'utility':growth-.1*dd,'decisions':summary}

def collect(policy,m,sampler,iteration):
 buffer=[];summaries=[];stds=[]
 for group in range(2):
  t,capital,key=sampler.choose(iteration*2+group);episodes=[rollout(policy,m,t,capital,sampler.rng) for _ in range(4)];rewards=np.array([info['utility'] for _,info in episodes]);stds.append(float(rewards.std()));advantages=(rewards-rewards.mean())/max(rewards.std(),.01)
  for (records,info),adv in zip(episodes,advantages):
   for row in records:row['advantage']=float(adv);buffer.append(row)
   summaries.append({'group':group,'regime':key,'advantage':float(adv),**info})
 return buffer,summaries,{'group_reward_std':stds,'zero_variance_groups':sum(s<1e-8 for s in stds)}

def optimize(policy,opt,buffer,rng):
 params=[p for p in policy.parameters() if p.requires_grad];logs=[];steps=0;identity=0.;stopped=False
 for epoch in range(2):
  order=rng.permutation(len(buffer));opt.zero_grad(set_to_none=True)
  for j,index in enumerate(order):
   row=buffer[int(index)];p=policy(row['encoded']);old=torch.as_tensor(row['old'],device=p.device,dtype=p.dtype);ref=torch.as_tensor(row['reference'],device=p.device,dtype=p.dtype)
   if steps==0:
    error=float(abs(p.detach()-old).max());identity=max(identity,error)
    if error>2e-6:raise AssertionError('Collection/update likelihood mismatch')
   loss,stats=clipped_loss(p,old,ref,row['actions'],row['advantage'])
   if stats['old_kl']>.03:opt.zero_grad(set_to_none=True);stopped=True;break
   if not torch.isfinite(loss):raise AssertionError('Nonfinite RL loss')
   (loss/4).backward();logs.append({'loss':float(loss.detach()),**stats})
   if (j+1)%4==0 or j==len(order)-1:
    torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True);opt.step();opt.zero_grad(set_to_none=True);steps+=1
  if stopped:break
 if not logs:raise AssertionError('No RL update')
 return {'optimizer_steps':steps,'likelihood_identity_error':identity,'early_stop_kl':stopped,**{k:float(np.mean([r[k] for r in logs])) for k in logs[0]}}
