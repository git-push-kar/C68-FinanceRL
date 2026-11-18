"""Full-path gradients through normalized candidate mixtures and exact fees."""
import numpy as np
import torch
from cash_execution import execute

def stock_step(weights,cash,target,rel):
    with torch.no_grad():
        reference=execute(weights.detach().numpy(),float(cash.detach()),target.detach().numpy(),rel.detach().numpy())
        sign=torch.sign((1-reference.trading_cost)*target-weights)
    cost=.001*(sign*(target-weights)).sum()/(1+.001*(sign*target).sum())
    values=(1-cost)*target*rel;growth=values.sum()
    # Candidate mixtures are fully invested. A constrained policy's target
    # derivatives sum to zero; no fictitious negative-cash position is allowed.
    return values/growth,weights.new_zeros(()),growth-1

def path_returns(probabilities,candidates,initial_weights,initial_cash,relatives):
    weights=torch.as_tensor(initial_weights,dtype=torch.float64);cash=torch.as_tensor(initial_cash,dtype=torch.float64)
    relatives=torch.as_tensor(relatives,dtype=torch.float64);candidates=torch.as_tensor(candidates,dtype=torch.float64)
    if probabilities.shape!=((len(relatives)+4)//5,5):raise ValueError('One five-way mixture per decision')
    result=[]
    for day,rel in enumerate(relatives):
        if day%5==0:
            p=probabilities[day//5];p=p/p.sum();target=p@candidates[day//5]
            weights,cash,net=stock_step(weights,cash,target,rel)
        else:
            values=weights*rel;growth=values.sum()+cash;weights,cash,net=values/growth,cash/growth,growth-1
        result.append(net)
    return torch.stack(result)
