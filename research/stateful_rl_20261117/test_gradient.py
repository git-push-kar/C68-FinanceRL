import numpy as np
import torch
from policy import expected_reward
from market import ACTIONS

def test_all_action_expectation_and_gradient_against_independent_sum():
 torch.manual_seed(4)
 logits=torch.randn(5,7,dtype=torch.float64,requires_grad=True)
 p=logits.softmax(-1);r=torch.linspace(-.1,.2,7**5,dtype=torch.float64)
 exact=expected_reward(p,r)
 joint=torch.ones(7**5,dtype=torch.float64)
 for asset in range(5):joint=joint*p[asset,torch.tensor(ACTIONS[:,asset],dtype=torch.long)]
 assert torch.allclose(exact,(joint*r).sum(),atol=1e-12)
 g=torch.autograd.grad(exact,logits,retain_graph=True)[0]
 independent=torch.autograd.grad((joint*r).sum(),logits)[0]
 assert torch.allclose(g,independent,atol=1e-12)
 assert torch.isfinite(g).all() and g.abs().sum()>0
