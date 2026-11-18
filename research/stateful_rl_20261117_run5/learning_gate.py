"""Balanced causal curriculum and disjoint-year conditional-learning gate."""
import numpy as np
import torch
from anchor import anchor
from market import execute

CAPITALS=[500.,5000.,50000.,100000.,200000.]

def dataset(m,seed,training=True):
 rng=np.random.default_rng(seed);first,last=m.bounds('2015-01-01' if training else '2023-01-01','2022-12-31' if training else '2023-12-31');first+=1;buckets={i:[] for i in range(7)}
 for i in range(6000 if training else 2400):
  t=int(rng.integers(first,last+1));capital=float(rng.choice(CAPITALS));w=np.zeros(5) if i%3==0 else rng.dirichlet(np.ones(6))[:5]
  cash,q,_,_=execute(capital,np.zeros(5,int),m.close[t],w);obs=m.observe(t,float(cash[0]),q[0],capital);synthetic=training and i%4==0
  if synthetic:
   # Counterfactual risk measurements teach numeric conditioning, never profits.
   for r in obs['stocks']:r['vol20']=float(rng.uniform(.08,.80))
  focus=int(rng.integers(5));action=int(anchor(obs,'inverse_vol')[focus]);buckets[action].append({'observation':obs,'focus':focus,'actions':[action],'permutation':rng.permutation(5).tolist(),'synthetic_risk':synthetic})
 rows=[]
 if training:
  if any(not rows for rows in buckets.values()):raise AssertionError('Missing training action class')
  for i in range(2048):
   bucket=buckets[i%7];rows.append(bucket[int(rng.integers(len(bucket)))])
  rng.shuffle(rows)
 else:
  for action,bucket in buckets.items():
   for i in rng.choice(len(bucket),min(16,len(bucket)),replace=False):rows.append(bucket[int(i)])
 return rows,{'class_pool_sizes':{str(k):len(v) for k,v in buckets.items()},'examples':len(rows),'training':training,'date_start':str(m.dates[first]),'date_end':str(m.dates[last]),'synthetic_risk_examples':sum(r['synthetic_risk'] for r in rows)}

def assess(policy,rows,encoded):
 matrix=np.zeros((7,7),int)
 with torch.inference_mode():
  for row,e in zip(rows,encoded):matrix[row['actions'][0],int(policy(e).argmax(1).item())]+=1
 totals=matrix.sum(1);eligible=totals>=8;recall=np.diag(matrix)/np.maximum(totals,1)
 return {'accuracy':float(np.trace(matrix)/matrix.sum()),'balanced_accuracy':float(recall[eligible].mean()),'minimum_eligible_recall':float(recall[eligible].min()),'recall':recall.tolist(),'confusion':matrix.tolist(),'eligible_classes':np.flatnonzero(eligible).tolist(),'distinct_predictions':int((matrix.sum(0)>0).sum()),'constant_accuracy':float(totals.max()/totals.sum()),'balanced_constant_accuracy':float(1/eligible.sum())}

def passes(result,raw):
 return bool(result['balanced_accuracy']>=.60 and result['balanced_accuracy']>=raw['balanced_accuracy']+.15 and result['minimum_eligible_recall']>=.20 and result['distinct_predictions']>=4)
