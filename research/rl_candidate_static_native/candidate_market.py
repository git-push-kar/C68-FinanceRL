"""Nifty strategy candidates with causal portfolio-level summaries."""
import numpy as np
from exposure_market import Market as CashMarket,tensor
from cash_execution import execute
from metrics import metrics_from_infos
from execution_policy import wait_for_compute_window

NAMES=['equal_weight','inverse_volatility','momentum_126_skip21','reversal_5']
SCALES=np.array([50,10,40,40,50,10]*4+[40,50,100],dtype=np.float32)

class Market(CashMarket):
    def observe(self):
        e=self.env;i=e.index-1
        if i not in self.cache:
            p=e.prices;returns=p[i-125:i+1]/p[i-126:i]-1
            vol=returns.std(0).clip(1e-5);momentum=p[i-21]/p[i-126]-1;short=p[i]/p[i-5]-1
            def tilt(x):
                z=(x-x.mean())/(x.std()+1e-8);w=np.exp(np.clip(z,-2,2));return w/w.sum()
            inv=1/vol;inv/=inv.sum()
            targets=np.stack([np.ones(e.n_assets)/e.n_assets,inv,tilt(momentum),tilt(-short)])
            targets=.5*targets+.5/e.n_assets
            if targets.max()>.30 or not np.allclose(targets.sum(1),1):raise AssertionError('Original candidate constraint failed')
            descriptive=[]
            for w in targets:
                history=returns@w
                descriptive.append([100*(w@momentum),100*(w@short),100*np.std(history[-20:])*np.sqrt(252),
                                    100*np.std(history)*np.sqrt(252),100*np.mean(history)*252])
            self.cache[i]=(np.asarray(descriptive),targets)
        descriptive,targets=self.cache[i]
        costs=np.array([10000*execute(e.weights,e.cash_weight,t,np.ones(e.n_assets)).trading_cost for t in targets])
        candidate=np.c_[descriptive,costs].reshape(-1)
        risk=[100*np.std(e.returns[-20:])*np.sqrt(252) if e.returns else 0.,100*(1-e.equity/e.peak_equity),100*e.cash_weight]
        summary=np.asarray([float(f'{x:.2f}') for x in np.r_[candidate,risk]],dtype=np.float32)
        return {'summary':summary,'candidate_targets':targets.copy()},targets

def evaluate(policy,market,device,fixed=None):
    import torch
    state=market.reset();probabilities=[]
    with torch.no_grad():
        while True:
            wait_for_compute_window();obs,targets=state
            if fixed is None:p=policy(tensor(obs,device))[0].probs[0].cpu().numpy().astype(float)
            elif isinstance(fixed,int):p=np.eye(4)[fixed]
            else:p=np.asarray(fixed,dtype=float)
            if p.shape!=(4,) or not np.isfinite(p).all() or (p<0).any() or p.sum()<=0:raise ValueError('Invalid candidate mixture')
            p/=p.sum();probabilities.append(p.tolist());state,_,done=market.step(p@targets)
            if done:break
    return {'metrics':metrics_from_infos(market.infos),'daily':market.infos,'probabilities':probabilities}
