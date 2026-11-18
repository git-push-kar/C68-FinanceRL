import numpy as np
import torch
from episodic import clipped_loss,WindowSampler
from market import Market
from learning import dataset,anchor

def test_positive_advantage_increases_selected_probability():
 logits=torch.zeros(49,7,requires_grad=True);p=logits.softmax(-1);old=p.detach().clone();ref=old.clone();actions=np.arange(49)%7
 loss,stats=clipped_loss(p,old,ref,actions,1.);loss.backward()
 assert stats['ratio']==1. and torch.all(logits.grad[torch.arange(49),torch.tensor(actions)]<0)
 assert torch.all(logits.grad.sum(1).abs()<1e-7)

def test_rl_windows_never_cross_training_boundary():
 m=Market();s=WindowSampler(m,42)
 for bucket in s.buckets.values():
  for t in bucket:assert str(m.dates[t+60])<'2024-01-01'

def test_curriculum_keeps_all_assets_and_only_past_dates():
 m=Market();rows=dataset(m,20,42,True);gate=dataset(m,10,43,False)
 assert all(len(r['actions'])==49 and r['observation']['universe']==m.tickers for r in rows+gate)
 assert all(r['observation']['asof']<'2023-01-01' for r in rows)
 assert all('2023-01-01'<=r['observation']['asof']<'2024-01-01' and not r['synthetic_risk'] for r in gate)
 assert len(set(np.array([r['actions'] for r in rows]).ravel()))>=4
