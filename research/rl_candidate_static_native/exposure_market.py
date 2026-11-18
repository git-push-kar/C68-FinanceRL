"""Causal, cash-aware five-session equity-exposure environment."""
from types import SimpleNamespace
import numpy as np
from env import PortfolioEnv
from cash_execution import execute
from metrics import metrics_from_infos
from execution_policy import wait_for_compute_window

LEVELS=np.array([.25,.5,.75,1.])
NAMES=['equity_25','equity_50','equity_75','equity_100']
SCALES=np.array([20,30,40,60,30,30,50,30,100,100,30,50,10,10,10,10],dtype=np.float32)

class Market:
    def __init__(self,data,start,end):
        p,v,t,d=PortfolioEnv._load_prices(data)
        keep=d<=np.datetime64(end);p=p[keep];d=d[keep]
        first=int(np.searchsorted(d,np.datetime64(start)));i=first-1
        if i<253 or first>=len(p):raise ValueError('Insufficient causal history')
        daily=(p[1:]/p[:-1]-1).mean(1)
        basket=np.r_[1.,np.cumprod(1+daily)]
        self.env=SimpleNamespace(prices=p,dates=d,tickers=t,start_index=i,n_assets=len(t),basket=basket)
        self.cache={};self.reset()
    def reset(self,index=None):
        e=self.env;e.index=e.start_index if index is None else int(index)
        e.weights=np.zeros(e.n_assets);e.cash_weight=1.;e.equity=e.peak_equity=100000.
        e.returns=[];self.infos=[]
        return self.observe()
    def observe(self):
        e=self.env;i=e.index-1
        if i not in self.cache:
            b=e.basket;p=e.prices;daily=b[1:i+1]/b[:i]-1
            history=[100*(b[i]/b[i-h]-1) for h in (20,60,126,252)]
            history += [100*np.std(daily[-h:])*np.sqrt(252) for h in (20,60)]
            history += [100*(1-b[i]/max(b[i-251:i+1])),100*(b[i]/np.mean(b[i-199:i+1])-1),100*np.mean(p[i]>p[i-126])]
            self.cache[i]=np.asarray(history)
        targets=LEVELS[:,None]*np.ones((4,e.n_assets))/e.n_assets
        costs=[10000*execute(e.weights,e.cash_weight,t,np.ones(e.n_assets)).trading_cost for t in targets]
        state=np.r_[self.cache[i],100*e.weights.sum(),100*np.std(e.returns[-20:])*np.sqrt(252) if e.returns else 0.,100*(1-e.equity/e.peak_equity),costs]
        # Exactly the numerical precision exposed to the native text policy.
        state=np.asarray([float(f'{x:.2f}') for x in state],dtype=np.float32)
        return {'summary':state,'candidate_targets':targets.astype(np.float32)},targets
    def step(self,target,max_days=5):
        if not isinstance(max_days,int) or not 1<=max_days<=5:raise ValueError('One to five sessions per step')
        e=self.env;reward=0.;signal=str(e.dates[e.index-1])[:10]
        for day in range(max_days):
            wait_for_compute_window()
            if e.index>=len(e.prices)-1:raise RuntimeError('Finished episode')
            decision=str(e.dates[e.index])[:10]
            actual=np.asarray(target,dtype=float) if day==0 else e.weights.copy()
            result=execute(e.weights,e.cash_weight,actual,e.prices[e.index+1]/e.prices[e.index])
            e.weights=result.stock_weights;e.cash_weight=result.cash_weight;e.index+=1
            e.equity*=1+result.net_return;e.peak_equity=max(e.peak_equity,e.equity);e.returns.append(result.net_return)
            utility=result.net_return-1.5*result.net_return**2;reward+=100*utility
            self.infos.append({'net_return':result.net_return,'gross_return':result.gross_return,
                'trading_cost':result.trading_cost,'turnover':result.turnover,'equity':e.equity,
                'drawdown':1-e.equity/e.peak_equity,'volatility':float(np.std(e.returns[-20:])),
                'reward':100*utility,'decision_date':decision,'return_date':str(e.dates[e.index])[:10],
                'signal_close_date':signal,'cash_weight':e.cash_weight})
            done=e.index>=len(e.prices)-1
            if done:break
        return (None if done else self.observe()),reward,done

def tensor(obs,device):
    import torch
    return {k:torch.as_tensor(v,device=device).unsqueeze(0) for k,v in obs.items()}

def evaluate(policy,market,device,fixed=None):
    import torch
    state=market.reset();probabilities=[]
    with torch.no_grad():
        while True:
            wait_for_compute_window();obs,targets=state
            if fixed is None:
                p=policy(tensor(obs,device))[0].probs[0].cpu().numpy().astype(float);p/=p.sum();target=p@targets
            else:
                exposure=float(fixed(obs) if callable(fixed) else fixed)
                if not .25<=exposure<=1.:raise ValueError('Control exposure outside bounds')
                p=np.array([np.nan]*4);target=np.ones(market.env.n_assets)*exposure/market.env.n_assets
            probabilities.append(p.tolist());state,_,done=market.step(target)
            if done:break
    metrics=metrics_from_infos(market.infos)
    metrics['annualized_quadratic_utility']=252*np.mean([i['reward']/100 for i in market.infos])
    return {'metrics':metrics,'daily':market.infos,'probabilities':probabilities if fixed is None else None}
