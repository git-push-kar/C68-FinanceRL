"""Matched cost/delay controls for the prespecified development sensitivity grid."""
import argparse
from pathlib import Path
import json
import numpy as np
from sklearn.covariance import LedoitWolf
from classical_panel import min_variance,hierarchical_risk_parity
from env import PortfolioEnv,FinanceRewardConfig
from execution_policy import wait_for_compute_window
from metrics import metrics_from_infos
from model import write_json


def episode(name,cost,lag,data):
    if lag not in (0,1):raise ValueError('Expected zero or one close of delay')
    env=PortfolioEnv(data,start_date='2023-01-01',end_date='2024-12-31',
        covariance_window=252,reward_config=FinanceRewardConfig(transaction_cost_bps=cost,rebalance_interval=21))
    env.reset();infos=[]
    while True:
        wait_for_compute_window();signal_index=env.index-lag
        if len(infos)%21==0:
            history=env.prices[signal_index-252:signal_index+1]
            if len(history)!=253:raise ValueError('Insufficient signal history')
            cov=LedoitWolf().fit(history[1:]/history[:-1]-1).covariance_
            weights=min_variance(cov) if name=='minimum_variance' else hierarchical_risk_parity(cov)
            action=np.log(np.maximum(weights,1e-30))
        else:action=np.zeros(env.n_assets) # ignored by the simulator between rebalances
        signal_date=str(env.dates[signal_index])[:10]
        _,_,done,truncated,info=env.step(action);info['signal_close_date']=signal_date
        if lag and not signal_date<info['decision_date']:raise AssertionError('Signal was not delayed')
        infos.append(info)
        if done or truncated:return {'metrics':metrics_from_infos(infos),'daily':infos}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True)
    p.add_argument('--data',default='../data/nifty50_prices.csv');args=p.parse_args()
    output=Path(args.output);output.mkdir(parents=True,exist_ok=False);summary={}
    for name in ('minimum_variance','hrp'):
        for cost,lag in ((0.,0),(5.,0),(10.,0),(20.,0),(10.,1)):
            key=f'{name}_cost{cost:g}_lag{lag}';result=episode(name,cost,lag,args.data)
            write_json(output/(key+'.json'),result);summary[key]=result['metrics'];print(key,json.dumps(result['metrics']),flush=True)
    write_json(output/'summary.json',{'results':summary,'role':'matched frozen-rule development sensitivities',
        'test2025_accessed':False,'no_parameter_or_policy_selection':True})


if __name__=='__main__':main()
