"""Piecewise differentiable version of the verified self-financing ledger."""
import numpy as np
import torch
from cash_execution import execute

def step(weights,cash,target,relatives,fee_bps=10.):
    # Locate the buy/sell region without differentiating the numerical solver.
    # Within that region, solve its linear fee equation exactly in Torch.
    with torch.no_grad():
        actual=execute(weights.detach().cpu().numpy(),float(cash.detach()),
            target.detach().cpu().numpy(),relatives.detach().cpu().numpy(),fee_bps)
        sign=torch.sign((1-actual.trading_cost)*target-weights)
    k=fee_bps/10000.
    cost=k*(sign*(target-weights)).sum()/(1+k*(sign*target).sum())
    stock=(1-cost)*target*relatives
    cash_value=(1-cost)*(1-target.sum())
    growth=stock.sum()+cash_value
    return stock/growth,cash_value/growth,growth-1,cost

def path_returns(exposures,initial_weights,initial_cash,relatives):
    weights=torch.as_tensor(initial_weights,dtype=torch.float64)
    cash=torch.as_tensor(initial_cash,dtype=torch.float64)
    relatives=torch.as_tensor(relatives,dtype=torch.float64)
    returns=[]
    if len(exposures)!=(len(relatives)+4)//5:raise ValueError('One exposure per five sessions')
    for day,rel in enumerate(relatives):
        if day%5==0:
            target=exposures[day//5]*torch.ones_like(weights)/len(weights)
            weights,cash,net,_=step(weights,cash,target,rel)
        else:
            stock=weights*rel;growth=stock.sum()+cash
            weights,cash,net=stock/growth,cash/growth,growth-1
        returns.append(net)
    return torch.stack(returns)

def negative_sharpe(returns):
    variance=returns.var(unbiased=False).clamp_min(1e-12)
    return -np.sqrt(252)*returns.mean()/torch.sqrt(variance)
