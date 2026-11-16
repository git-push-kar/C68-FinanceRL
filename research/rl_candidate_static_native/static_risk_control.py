"""Fixed initial-risk allocation, matched monthly execution and costs.

The single covariance estimate uses only information at the first decision
close of the requested evaluation period. No future observation changes the
target; actual holdings drift and are rebalanced to that fixed target.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from sklearn.covariance import LedoitWolf
from classical_panel import min_variance
from env import PortfolioEnv,FinanceRewardConfig
from execution_policy import wait_for_compute_window
from metrics import metrics_from_infos
from model import write_json


def episode(start,end,cost_bps,data,signal_lag=0):
    if signal_lag not in (0,1):raise ValueError('Expected zero or one close of signal delay')
    env=PortfolioEnv(data,start_date=start,end_date=end,covariance_window=252,
        reward_config=FinanceRewardConfig(transaction_cost_bps=cost_bps,rebalance_interval=21))
    env.reset();signal_index=env.index-signal_lag;history=env.prices[signal_index-252:signal_index+1]
    cov=LedoitWolf().fit(history[1:]/history[:-1]-1).covariance_
    target=min_variance(cov);action=np.log(np.maximum(target,1e-30))
    signal_date=str(env.dates[signal_index])[:10];infos=[]
    while True:
        wait_for_compute_window();_,_,done,truncated,info=env.step(action)
        info['fixed_risk_signal_close_date']=signal_date;infos.append(info)
        if done or truncated:break
    return {'metrics':metrics_from_infos(infos),'daily':infos,'fixed_target':target.tolist(),
        'signal_close_date':signal_date,'role':'known initial 252-return risk allocation, fixed target with 21-session rebalancing'}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True)
    p.add_argument('--data',default='../data/nifty50_prices.csv');args=p.parse_args()
    output=Path(args.output);output.mkdir(parents=True,exist_ok=False)
    result=episode('2023-01-01','2024-12-31',10.,args.data)
    write_json(output/'result.json',result);print(json.dumps(result['metrics']),flush=True)


if __name__=='__main__':main()
