import copy
import numpy as np
import torch
from torch import nn
from transformers import AutoTokenizer
from research.counterfactual_rl.market import Market,project_scores,LEVELS,execute
from research.counterfactual_rl.policy import NativePolicy
from research.counterfactual_rl.learning import counterfactuals,objective,Sampler

def test_projection_caps_and_budget():
 for n in [0,1,3,10,49]:
  a=np.zeros(49);a[:n]=np.arange(1,n+1);w=project_scores(a)
  assert np.isclose(w.sum(),min(1,n*.1)) and w.max()<=.10000001 and (w[a==0]==0).all()

def test_residual_allocator_uses_affordable_lots_without_buying_zero_targets():
 price=np.full(49,103.);c,q,*_=execute(100000,np.zeros(49,int),price,np.ones(49)/49)
 assert c<103*2 and q.sum()>950
 w=np.zeros(49);w[:10]=.1;c,q,*_=execute(100000,np.zeros(49,int),price,w)
 assert q[10:].sum()==0

def test_canonical_input_is_identical_after_reordering():
 p=NativePolicy.__new__(NativePolicy);nn.Module.__init__(p);p.max_tokens=0;p.tokenizer=AutoTokenizer.from_pretrained('C:/Users/Parth/Documents/PolyvalentFinance/local_models/InternVL3-2B',trust_remote_code=True,local_files_only=True,use_fast=True)
 m=Market();t,_=m.bounds('2023-01-01','2023-12-31');obs=m.observe(t,100000,np.zeros(49,int),100000);other=copy.deepcopy(obs);other['stocks'].reverse();other['universe'].reverse()
 a=p.encode(obs);b=p.encode(other,permutation=np.arange(49)[::-1]);assert torch.equal(a['ids'],b['ids']);assert np.array_equal(a['numeric_state'],b['numeric_state']);assert a['prompt_sha256']==b['prompt_sha256']

def test_counterfactual_credit_and_training_boundary():
 m=Market();sampler=Sampler(m,42);t,c=sampler.start();assert str(m.dates[t+60])<'2023-01-01';q=np.zeros(49,int);pending=np.zeros(2);actions=np.full(49,3)
 reward,r=counterfactuals(m,t,c,q,pending,actions,t+20)
 assert reward.shape==(49,7) and np.isfinite(reward).all();assert np.allclose(reward[:,3],reward[0,3])
 logits=torch.zeros(49,7,requires_grad=True);p=logits.softmax(1);values=np.zeros((49,7));values[0,6]=.1;loss,_=objective(p,p.detach(),p.detach(),values);loss.backward()
 assert logits.grad[0,6]<0 and torch.all(logits.grad[1:]==0)
