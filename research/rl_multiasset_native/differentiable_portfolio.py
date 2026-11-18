"""Autograd counterpart of the existing capped, cost-aware simulator.

No model or market data are loaded on import. Financial arithmetic is float64;
neural parameters remain FP32. Boundary active sets are piecewise differentiable.
"""
import torch


def capped_weights(scores,cap):
    scores=scores.to(torch.float64)
    if not 0<cap<=1 or len(scores)*cap<1:
        raise ValueError('Infeasible portfolio cap')
    free=torch.ones_like(scores,dtype=torch.bool)
    fixed=torch.zeros_like(scores)
    for _ in range(len(scores)):
        active_scores=scores[free]
        mass=torch.where(free,scores-active_scores.max(),torch.full_like(scores,-torch.inf)).exp()
        candidate=(1-fixed.sum())*mass/mass.sum()
        over=free & (candidate>cap)
        if not bool(over.any()):return fixed+candidate
        fixed=fixed+over.to(scores.dtype)*cap
        free=free & ~over
    return fixed


def portfolio_step(target,holdings,relatives,cost_bps):
    turnover=(target-holdings).abs().sum()
    cost=turnover*cost_bps/10000
    gross=(target*(relatives-1)).sum()
    net=(1-cost)*(1+gross)-1
    return net,target*relatives/(1+gross),turnover


def replay(scores,relatives,interval=21,cap=.3,cost_bps=10.):
    """Replay fixed decision scores with graph intact; useful for parity tests."""
    holdings=torch.zeros_like(scores[0],dtype=torch.float64)
    results=[]
    for day,relative in enumerate(relatives):
        target=capped_weights(scores[day//interval],cap) if day%interval==0 else holdings
        net,holdings,_=portfolio_step(target,holdings,relative,cost_bps)
        results.append(net)
    return torch.stack(results)
