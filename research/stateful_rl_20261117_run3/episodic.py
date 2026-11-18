"""Critic-free clipped group-relative policy updates on actual account episodes."""
import numpy as np
import torch
from market import targets,HORIZON

def group_advantages(rewards):
 rewards=np.asarray(rewards,float)
 return (rewards-rewards.mean())/max(float(rewards.std()),.01)

def clipped_loss(probabilities,old,reference,actions,advantage):
 indices=torch.as_tensor(actions,dtype=torch.long,device=probabilities.device)
 logp=probabilities.gather(1,indices[:,None]).log().sum()
 oldlog=old.gather(1,indices[:,None]).log().sum()
 ratio=torch.exp((logp-oldlog).clamp(-20,20))
 advantage=probabilities.new_tensor(float(advantage))
 surrogate=torch.minimum(ratio*advantage,ratio.clamp(.8,1.2)*advantage)
 reference_kl=(probabilities*(probabilities.log()-reference.log())).sum(-1).mean()
 old_kl=(old*(old.log()-probabilities.log())).sum(-1).mean()
 entropy=-(probabilities*probabilities.log()).sum(-1).mean()
 return -surrogate+.01*reference_kl-.003*entropy,{'ratio':float(ratio.detach()),'reference_kl':float(reference_kl.detach()),'old_kl':float(old_kl.detach()),'entropy':float(entropy.detach())}

class WindowSampler:
 def __init__(self,market,seed):
  self.market=market;self.rng=np.random.default_rng(seed);first,last=market.bounds('2015-01-01','2023-12-31');self.buckets={}
  for t in range(first,last-60+1,5):
   trend=float(np.mean(market.index[t]/market.index[t-60]-1))
   daily=(market.relative[t-19:t+1]-1).mean(1);vol=float(daily.std()*np.sqrt(252))
   key=('up' if trend>=0 else 'down')+('_highvol' if vol>=.20 else '_lowvol')
   self.buckets.setdefault(key,[]).append(t)
  self.keys=sorted(self.buckets)
 def choose(self,number):
  key=self.keys[number%len(self.keys)];t=int(self.rng.choice(self.buckets[key]));capital=float(self.rng.choice([500,5000,50000,100000,200000]))
  return t,capital,key

def rollout(policy,market,t,capital,rng):
 cash=float(capital);qty=np.zeros(5,np.int64);values=[];experience=[];summary=[];begin=t
 for decision in range(3):
  obs=market.observe(t,cash,qty,capital);encoded=policy.encode(obs,permutation=rng.permutation(5))
  with torch.inference_mode():
   probabilities=policy(encoded);actions=torch.distributions.Categorical(probs=probabilities).sample().cpu().numpy()
   old=probabilities.double().cpu().numpy()
   with policy.reference_context():reference=policy(encoded).double().cpu().numpy()
  w=targets(actions,qty,cash,market.close[t]);result=market.horizon(t,cash,qty,w,t+HORIZON,keep=actions==0)
  experience.append({'encoded':encoded,'old':old,'reference':reference,'actions':actions,'observation':obs})
  summary.append({'signal_date':obs['asof'],'actions':actions.tolist(),'trade_shares':result['trade_shares'][0].tolist(),'prompt_sha256':encoded['prompt_sha256'],'tokens':encoded['ids'].shape[1]})
  cash=float(result['cash'][0]);qty=result['qty'][0];values.extend(result['values'][:,0].tolist());t+=HORIZON
 path=np.r_[capital,values];drawdown=float((1-path/np.maximum.accumulate(path)).max());growth=float(np.log(path[-1]/capital));utility=growth-.1*drawdown
 return experience,{'begin':str(market.dates[begin]),'end':str(market.dates[t]),'initial':capital,'end_equity':float(path[-1]),'net_return':float(path[-1]/capital-1),'drawdown':drawdown,'utility':utility,'decisions':summary}

def collect(policy,market,sampler,iteration,groups=2):
 buffer=[];summaries=[];variances=[]
 for group in range(groups):
  t,capital,key=sampler.choose(iteration*groups+group);episodes=[]
  for member in range(4):
   records,info=rollout(policy,market,t,capital,sampler.rng);episodes.append((records,info))
  rewards=[info['utility'] for _,info in episodes];advantages=group_advantages(rewards);variances.append(float(np.std(rewards)))
  for (records,info),advantage in zip(episodes,advantages):
   for record in records:record['advantage']=float(advantage);buffer.append(record)
   summaries.append({'group':group,'regime_bucket':key,'advantage':float(advantage),**info})
 return buffer,summaries,{'zero_variance_groups':int(sum(v<1e-8 for v in variances)),'group_reward_std':variances}

def optimize(policy,optimizer,buffer,rng,epochs=2):
 parameters=[p for p in policy.parameters() if p.requires_grad];logs=[];steps=0;identity_error=0.;stopped=False;norm=0.
 for epoch in range(epochs):
  order=rng.permutation(len(buffer));optimizer.zero_grad(set_to_none=True)
  for j,index in enumerate(order):
   row=buffer[int(index)];p=policy(row['encoded']);old=torch.tensor(row['old'],dtype=p.dtype,device=p.device);ref=torch.tensor(row['reference'],dtype=p.dtype,device=p.device)
   if epoch==0:
    # Before the FIRST optimizer step, every likelihood must match collection.
    if steps==0:
     error=float((p.detach()-old).abs().max());identity_error=max(identity_error,error)
     if error>2e-6:raise AssertionError('Rollout/update kernel likelihood mismatch')
   loss,stats=clipped_loss(p,old,ref,row['actions'],row['advantage'])
   if not torch.isfinite(loss):raise AssertionError('Nonfinite clipped policy loss')
   if stats['old_kl']>.05:
    optimizer.zero_grad(set_to_none=True);stopped=True;break
   (loss/4).backward();logs.append({'loss':float(loss.detach()),**stats})
   if (j+1)%4==0 or j==len(order)-1:
    norm=float(torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True));optimizer.step();optimizer.zero_grad(set_to_none=True);steps+=1
  if stopped:break
 if not logs:raise AssertionError('No policy-update examples processed')
 result={name:float(np.mean([r[name] for r in logs])) for name in logs[0]}
 return {'optimizer_steps':steps,'likelihood_identity_error':identity_error,'last_gradient_norm':norm,'early_stop_update_kl':stopped,**result}
