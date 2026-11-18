"""Causal inverse-volatility curriculum. No teacher is used by the learned policy."""
import copy
import numpy as np
import torch
from market import LEVELS,execute

CAPITALS=[500.,5000.,50000.,100000.,200000.]

def anchor(obs):
 rows=obs['stocks'];eq=obs['equity'];prices=np.array([r['historical_quote'] for r in rows]);vol=np.maximum([r['vol20'] for r in rows],.03)
 # Only assets for which one whole share fits the10percent cap can be bought.
 feasible=prices*1.0015005<=eq*.10
 score=feasible/vol;weights=np.minimum(score/max(score.sum(),1e-12),.10)
 labels=abs(weights[:,None]-LEVELS).argmin(1)
 # A positive target that cannot buy one share becomes cash, not a fictional lot.
 labels=np.where(LEVELS[labels]*eq>=prices*1.0015005,labels,0)
 return labels.astype(int)

def dataset(m,n,seed,training):
 rng=np.random.default_rng(seed);first,last=m.bounds('2018-07-02' if training else '2023-01-01','2022-12-31' if training else '2023-12-31');rows=[]
 for k in range(n):
  t=int(rng.integers(first,last+1));capital=float(rng.choice(CAPITALS));w=np.zeros(49) if k%3==0 else rng.dirichlet(np.full(50,.7))[:49]
  c,q,*_=execute(capital,np.zeros(49,int),m.close[t],w);obs=m.observe(t,c,q,capital)
  synthetic=training and k%4==0
  if synthetic:
   for s in obs['stocks']:s['vol20']=float(np.exp(rng.uniform(np.log(.03),np.log(1.2))))
  rows.append({'observation':obs,'actions':anchor(obs).tolist(),'permutation':rng.permutation(49).tolist(),'synthetic_risk':synthetic})
 return rows

def assess(policy,rows,encoded):
 matrix=np.zeros((7,7),int)
 with torch.inference_mode():
  for row,e in zip(rows,encoded):
   pred=policy(e).argmax(1).cpu().numpy();np.add.at(matrix,(np.array(row['actions']),pred),1)
 totals=matrix.sum(1);eligible=totals>=15;recall=np.diag(matrix)/np.maximum(totals,1)
 return {'accuracy':float(np.trace(matrix)/matrix.sum()),'balanced_accuracy':float(recall[eligible].mean()),'minimum_eligible_recall':float(recall[eligible].min()),'recall':recall.tolist(),'confusion':matrix.tolist(),'eligible_classes':np.flatnonzero(eligible).tolist(),'distinct_predictions':int((matrix.sum(0)>0).sum()),'constant_accuracy':float(totals.max()/totals.sum()),'balanced_constant_accuracy':float(1/eligible.sum())}

def passes(score,raw):
 return score['balanced_accuracy']>=.60 and score['balanced_accuracy']>=raw['balanced_accuracy']+.15 and score['minimum_eligible_recall']>=.20 and score['distinct_predictions']>=4
