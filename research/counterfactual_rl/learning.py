"""Training-only teacher choice and simulator-exact counterfactual financial RL."""
import numpy as np
import torch
from .market import LEVELS,project_scores,evaluate,execute

KINDS=['inverse_vol','momentum','risk_adjusted_momentum']
CAPITALS=[5000.,50000.,100000.,200000.]

def scores(obs,kind):
 r=obs['stocks'];vol=np.maximum([s['vol20'] for s in r],.05);trend=np.array([s['return120'] for s in r]);rank=np.argsort(np.argsort(trend,kind='stable'),kind='stable')/(len(r)-1)
 if kind=='inverse_vol':score=np.median(vol)/vol
 elif kind=='momentum':score=2**(4*(rank-.5))
 elif kind=='risk_adjusted_momentum':score=2**(4*(rank-.5))*np.median(vol)/vol
 else:raise ValueError(kind)
 return np.clip(score,.25,8.)

def teacher(obs,kind):
 score=scores(obs,kind)
 # Relative-score vocabulary; do not learn a hidden future price forecaster.
 return abs(np.log2(score[:,None])-np.log2(LEVELS[None,1:])).argmin(1)+1

def select_teacher(m):
 rows=[]
 for kind in KINDS:
  for year in range(2019,2023):
   result=evaluate(m,lambda o:(teacher(o,kind),{'teacher':kind}),f'{year}-01-01',f'{year}-12-31',100000.)
   rows.append({'kind':kind,'year':year,**result['metrics']})
 score={kind:float(np.mean([np.log1p(r['return'])-.1*r['max_drawdown'] for r in rows if r['kind']==kind])) for kind in KINDS}
 return {'selected':max(KINDS,key=lambda k:score[k]),'scores':score,'rows':rows,'criterion':'Mean annual log growth minus0.1drawdown in2019-2022 only. No2023/2024 validation or2025-2026 test selection.'}

def dataset(m,n,seed,kind,gate=False):
 rng=np.random.default_rng(seed);first,last=m.bounds('2022-01-01' if gate else '2018-07-02','2022-12-31' if gate else '2021-12-31');rows=[]
 for j in range(n):
  t=int(rng.integers(first,last+1));capital=float(rng.choice(CAPITALS));w=np.zeros(49) if j%3==0 else project_scores(rng.lognormal(0,1,49));c,q,*_=execute(capital,np.zeros(49,int),m.close[t],w);obs=m.observe(t,c,q,capital)
  synthetic=not gate and j%4==0
  if synthetic:
   for r in obs['stocks']:
    r['vol20']=float(np.exp(rng.uniform(np.log(.05),np.log(.8))));r['return120']=float(rng.uniform(-.5,1.))
  rows.append({'observation':obs,'actions':teacher(obs,kind).tolist(),'synthetic':synthetic})
 return rows

def assess(policy,rows,encoded):
 matrix=np.zeros((7,7),int);errors=[];uniform_errors=[]
 with torch.no_grad():
  for row,e in zip(rows,encoded):
   a=policy(e).argmax(1).cpu().numpy();labels=np.array(row['actions']);np.add.at(matrix,(labels,a),1)
   target=project_scores(LEVELS[labels]);errors.append(abs(project_scores(LEVELS[a])-target).sum());uniform_errors.append(abs(np.full(49,1/49)-target).sum())
 counts=matrix.sum(1);eligible=counts>=15;recall=np.diag(matrix)/np.maximum(counts,1)
 return {'accuracy':float(np.trace(matrix)/matrix.sum()),'balanced_accuracy':float(recall[eligible].mean()),'minimum_recall':float(recall[eligible].min()),'eligible_classes':np.flatnonzero(eligible).tolist(),'confusion':matrix.tolist(),'allocation_l1':float(np.mean(errors)),'uniform_allocation_l1':float(np.mean(uniform_errors))}

def passes(score,raw):
 return score['balanced_accuracy']>=.6 and score['minimum_recall']>=.2 and score['balanced_accuracy']>=raw['balanced_accuracy']+.15 and score['allocation_l1']<score['uniform_allocation_l1']*.8

class Sampler:
 def __init__(self,m,seed):
  self.rng=np.random.default_rng(seed);self.first,self.last=m.bounds('2018-07-02','2022-12-31');self.starts=np.arange(self.first,self.last-60,5)
 def start(self):return int(self.rng.choice(self.starts)),float(self.rng.choice(CAPITALS))

def utility(result,initial):
 path=np.r_[initial,result['values']];dd=float((1-path/np.maximum.accumulate(path)).max())
 return float(np.log(path[-1]/initial)-.1*dd)

def counterfactuals(m,t,c,q,pending,actions,end):
 """Marginalize every action for each stock with the other48 decisions fixed.

 This is a20-session receding-horizon RL objective, not a full long-horizon
 value estimator or a reproduction of COMA. Exact historical simulator rewards
 replace a learned critic. Only training-window future prices enter rewards.
 """
 actions=np.asarray(actions,int);equity=m.observe(t,c,q,1.,pending)['equity'];u=np.empty((49,7));common=m.horizon(t,c,q,project_scores(LEVELS[actions]),end,pending);base=utility(common,equity)
 for i in range(49):
  for a in range(7):
   if a==actions[i]:u[i,a]=base;continue
   changed=actions.copy();changed[i]=a;r=m.horizon(t,c,q,project_scores(LEVELS[changed]),end,pending);u[i,a]=utility(r,equity)
 return u,common

def objective(prob,old,reference,rewards):
 # Exact conditional categorical expectation: no sampled global shared reward,
 # no per-coordinate std scaling, no fictitious labels from validation/test.
 q=torch.as_tensor(rewards,dtype=prob.dtype,device=prob.device)
 advantage=(q-(old*q).sum(1,keepdim=True))/.01
 expected=(prob*advantage).sum(1).mean()
 kl=(prob*(prob.log()-reference.log())).sum(1).mean()
 loss=-expected+.02*kl
 return loss,{'expected_advantage':float(expected.detach()),'reference_kl':float(kl.detach()),'reward_spread':float((q.max(1).values-q.min(1).values).mean())}
