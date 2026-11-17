import numpy as np
import torch
from episodic import group_advantages,clipped_loss,WindowSampler
from market import Market

def test_group_baseline_is_shift_invariant_and_zero_variance_safe():
 r=np.array([-.03,.01,.05,.02]);assert np.allclose(group_advantages(r),group_advantages(r+100))
 assert np.array_equal(group_advantages([0,0,0,0]),np.zeros(4))
 assert np.isfinite(group_advantages([1,1,1,1])).all()

def test_unit_likelihood_ratio_and_nonzero_finite_policy_gradient():
 logits=torch.zeros(5,7,requires_grad=True);p=logits.softmax(-1);old=p.detach().clone();ref=old.clone()
 loss,stats=clipped_loss(p,old,ref,np.array([1,2,3,4,5]),1.)
 assert abs(stats['ratio']-1)<1e-6 and stats['old_kl']==0
 loss.backward();assert torch.isfinite(logits.grad).all() and logits.grad.abs().sum()>0

def test_regime_window_sampling_stays_inside_training_period():
 m=Market();s=WindowSampler(m,42);_,last=m.bounds('2015-01-01','2023-12-31')
 assert len(s.keys)>=3
 for i in range(100):
  t,capital,key=s.choose(i);assert t+60<=last and m.dates[t]>=np.datetime64('2014-12-31')
  assert capital in [500,5000,50000,100000,200000] and key in s.keys
