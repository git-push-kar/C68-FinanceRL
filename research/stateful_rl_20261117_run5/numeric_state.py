"""Representation control only: small numerical classifier, not final adapter/RL.

Uses exactly revision4 curriculum/gate states. No price forecasting, trading
evaluation or test-period access. Tests whether the conditional labels can be
learned from continuous features without a decimal-text interface.
"""
import sys,json
from pathlib import Path
import numpy as np
import torch
from torch import nn

def numeric_features(obs,focus):
 rows=obs['stocks'];s=rows[focus];eq=max(obs['equity'],1.);vol=np.maximum([r['vol20'] for r in rows],.01)
 own=[s['return5']/.1,s['return20']/.2,s['return60']/.4,s['return120']/.6,s['drawdown60']/.4,s['vol20']/.4,np.log(max(s['volume_ratio'],.01)),s['weight'],s['price']/eq,np.log1p(s['shares'])/8,s['affordable_shares_at_signal']*s['price']/eq,s['vol20']/vol.mean()]
 account=[obs['cash']/eq,np.log(eq)/12,np.log(max(obs['initial'],1.))/12,eq/max(obs['initial'],1.)-1,obs['max_stock_weight'],obs['max_sector_weight']]
 # Focus first, then sorted other rows: this encoder carries no ticker identity.
 others=sorted([i for i in range(5) if i!=focus],key=lambda i:rows[i]['vol20'])
 context=[]
 for i in [focus]+others:
  r=rows[i];context.extend([r['vol20']/.4,r['price']/eq,r['weight'],r['return60']/.4])
 return np.clip(np.asarray(own+account+context,np.float32),-10,10)

def score(model,x,y):
 with torch.no_grad():pred=model(x).argmax(1)
 counts=torch.bincount(y,minlength=7);correct=torch.bincount(y[pred==y],minlength=7);eligible=counts>=8;recall=correct/counts.clamp(min=1)
 return {'accuracy':float((pred==y).float().mean()),'balanced_accuracy':float(recall[eligible].mean()),'minimum_recall':float(recall[eligible].min()),'recall':recall.tolist()}

def main(run,out):
 torch.set_num_threads(2);torch.manual_seed(42);run=Path(run);rows=json.loads((run/'curriculum.json').read_text());gate=json.loads((run/'learning_gate_examples.json').read_text())
 def tensors(rows):return torch.tensor(np.stack([numeric_features(r['observation'],r['focus']) for r in rows])),torch.tensor([r['actions'][0] for r in rows])
 x,y=tensors(rows);gx,gy=tensors(gate);mean=x.mean(0);std=x.std(0).clamp(min=.05);x=(x-mean)/std;gx=(gx-mean)/std
 model=nn.Sequential(nn.Linear(x.shape[1],64),nn.Tanh(),nn.Linear(64,64),nn.Tanh(),nn.Linear(64,7));optimizer=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.01);history=[]
 for epoch in range(201):
  if epoch%25==0:history.append({'epoch':epoch,'train':score(model,x,y),'gate':score(model,gx,gy)})
  if epoch==200:break
  for indices in torch.randperm(len(x)).split(128):
   optimizer.zero_grad();loss=nn.functional.cross_entropy(model(x[indices]),y[indices]);loss.backward();optimizer.step()
 result={'scope':'Numerical representation control only, not InternVL or RL; fixed200epochs, not selected for test profits. Same training/gate examples and causal features; no ticker identity or future outcomes. Cannot substitute these scores for adapter learning.','feature_dimensions':x.shape[1],'history':history}
 Path(out).write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

if __name__=='__main__':main(sys.argv[1],sys.argv[2])
