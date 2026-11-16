"""Declared causal baselines on the exact finance simulator and date block."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf

from allocation import capped_softmax
from env import PortfolioEnv, FinanceRewardConfig
from execution_policy import wait_for_compute_window
from metrics import metrics_from_infos
from warmup_finance import risk_rule_scores


def min_variance(cov, cap=.3):
    n=len(cov)
    scale=max(float(np.trace(cov)/n),1e-12)
    scaled=cov/scale
    result=minimize(lambda w: float(w@scaled@w),np.full(n,1/n),
        jac=lambda w: 2*scaled@w,method='SLSQP',bounds=[(0,cap)]*n,
        constraints={'type':'eq','fun':lambda w:w.sum()-1,'jac':lambda w:np.ones(n)},
        options={'ftol':1e-10,'maxiter':500})
    if not result.success:
        raise RuntimeError('Minimum-variance optimization failed: '+result.message)
    return capped_softmax(np.log(np.maximum(result.x,1e-30)),cap)


def cluster_variance(cov,indices):
    sub=cov[np.ix_(indices,indices)]
    w=1/np.maximum(np.diag(sub),1e-12);w/=w.sum()
    return float(w@sub@w)


def hierarchical_risk_parity(cov,cap=.3):
    # Lopez de Prado (2016): single-linkage ordering and recursive bisection.
    std=np.sqrt(np.maximum(np.diag(cov),1e-12))
    corr=np.clip(cov/np.outer(std,std),-1,1)
    distance=np.sqrt(np.maximum((1-corr)/2,0))
    np.fill_diagonal(distance,0)
    order=leaves_list(linkage(squareform((distance+distance.T)/2,checks=False),method='single'))
    weights=np.ones(len(cov));clusters=[order.tolist()]
    while clusters:
        children=[]
        for cluster in clusters:
            if len(cluster)<=1:continue
            midpoint=len(cluster)//2
            left,right=cluster[:midpoint],cluster[midpoint:]
            vl,vr=cluster_variance(cov,left),cluster_variance(cov,right)
            alpha=vr/max(vl+vr,1e-20)
            weights[left]*=alpha;weights[right]*=1-alpha
            children.extend([left,right])
        clusters=children
    return capped_softmax(np.log(np.maximum(weights,1e-30)),cap)


def episode(name,start,end,cost_bps,data=None,signal_lag=0):
    data=data or Path(__file__).resolve().parent.parent/'data/nifty50_prices.csv'
    if signal_lag not in (0,1):raise ValueError('Expected zero or one close of signal delay')
    rate=.1 if name in ('slow_uniform','inverse_vol20_slow') else 1.
    interval=(21 if name in ('minimum_variance252_monthly','hrp252_monthly') else
              1_000_000 if name=='equal_weight_buy_hold' else 1)
    env=PortfolioEnv(data,start_date=start,end_date=end,
        reward_config=FinanceRewardConfig(transaction_cost_bps=cost_bps,rebalance_rate=rate,
                                         rebalance_interval=interval))
    obs,_=env.reset();infos=[];allocations=[];rebalance_dates=[];previous=None
    if signal_lag:
        env.index-=1;previous=env._observation();env.index+=1
    while True:
        wait_for_compute_window()
        t=len(infos)
        signal=previous if signal_lag else obs
        if name in ('equal_weight_daily','slow_uniform') or (name=='equal_weight_buy_hold' and t==0):
            action=np.zeros(env.n_assets)
        elif name=='inverse_vol20_slow':action=risk_rule_scores(signal['market'])
        elif name=='equal_weight_buy_hold' or t%21:
            action=np.log(np.maximum(env.weights,1e-30))
        else:
            signal_index=env.index-signal_lag
            history=env.prices[max(0,signal_index-252):signal_index+1]
            returns=history[1:]/history[:-1]-1
            if len(returns)<252:raise ValueError('252 causal historical returns required')
            cov=LedoitWolf().fit(returns).covariance_
            weights=min_variance(cov) if name=='minimum_variance252_monthly' else hierarchical_risk_parity(cov)
            action=np.log(np.maximum(weights,1e-30))
            rebalance_dates.append(str(env.dates[env.index])[:10])
        signal_date=str(env.dates[env.index-signal_lag])[:10];previous=obs
        obs,_,done,truncated,info=env.step(action);info['signal_close_date']=signal_date
        infos.append(info);allocations.append(env.weights.tolist())
        if done or truncated:break
    return {'metrics':metrics_from_infos(infos),'daily':infos,'drifted_weights':allocations,
            'scheduled_rebalance_dates':rebalance_dates,'rebalance_rate':rate,'signal_lag_closes':signal_lag}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True)
    p.add_argument('--start',default='2023-01-01')
    p.add_argument('--end',default='2024-12-31')
    p.add_argument('--cost-bps',type=float,default=10.)
    p.add_argument('--data',default=str(Path(__file__).resolve().parent.parent/'data/nifty50_prices.csv'))
    args=p.parse_args()
    if args.end>'2024-12-31':raise ValueError('Development panel refuses reserved/future evaluation')
    output=Path(args.output);output.mkdir(parents=True,exist_ok=False)
    names=['equal_weight_daily','equal_weight_buy_hold','slow_uniform','inverse_vol20_slow',
           'minimum_variance252_monthly','hrp252_monthly']
    protocol={'period':[args.start,args.end],'cost_bps':args.cost_bps,'strategies':names,
        'common':'same 49-stock fixed panel, return boundaries, actual drift/turnover, 0.30 target cap',
        'differences':'risk-rule/slow-uniform rebalance .1 daily; covariance strategies use 252 known returns and rebalance fully every 21 trading days',
        'source':'HRP: Lopez de Prado 2016, single linkage and recursive bisection; LW shrinkage + constrained SLSQP minimum variance',
        'role':'all declared development controls, not trained adapters or a new untouched test'}
    (output/'protocol.json').write_text(json.dumps(protocol,indent=2))
    summary={}
    for name in names:
        result=episode(name,args.start,args.end,args.cost_bps,data=args.data)
        (output/(name+'.json')).write_text(json.dumps(result,indent=2,allow_nan=False))
        summary[name]=result['metrics']
        print(name,json.dumps(result['metrics']),flush=True)
    (output/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False))


if __name__=='__main__':main()
