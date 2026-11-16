"""Check feasibility and KKT residuals of the numerical risk reference."""
import argparse
from pathlib import Path
import json
import numpy as np
from classical_panel import min_variance
from env import PortfolioEnv,FinanceRewardConfig
from model import write_json
from warmup_finance import label_indices


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True)
    p.add_argument('--data',default='../data/nifty50_prices.csv');args=p.parse_args()
    output=Path(args.output)
    if output.exists():raise FileExistsError(output)
    env=PortfolioEnv(args.data,end_date='2022-12-31',covariance_window=252,
        reward_config=FinanceRewardConfig(rebalance_interval=21));env.reset()
    _,holdout,_=label_indices(len(env.prices),252,0);rows=[]
    for index in holdout[np.linspace(0,len(holdout)-1,32,dtype=int)]:
        env.index=int(index);cov=env._observation()['covariance'].astype(np.float64)
        w=min_variance(cov);scaled=cov/(np.trace(cov)/len(cov));gradient=2*scaled@w
        free=(w>1e-6)&(w<.3-1e-6);level=float(gradient[free].mean())
        violations=[float(np.max(np.abs(gradient[free]-level)))]
        if (w<=1e-6).any():violations.append(float(np.maximum(level-gradient[w<=1e-6],0).max()))
        if (w>=.3-1e-6).any():violations.append(float(np.maximum(gradient[w>=.3-1e-6]-level,0).max()))
        rows.append({'date':str(env.dates[env.index])[:10],'sum_error':abs(float(w.sum())-1),
            'minimum_weight':float(w.min()),'maximum_weight':float(w.max()),
            'covariance_minimum_eigenvalue':float(np.linalg.eigvalsh(cov).min()),
            'scaled_KKT_violation':max(violations),'variance_to_uniform':float(w@cov@w/cov.mean())})
    maximum=max(row['scaled_KKT_violation'] for row in rows)
    passed=maximum<1e-4 and all(r['sum_error']<1e-10 and r['minimum_weight']>=0 and
        r['maximum_weight']<=.3+1e-10 and r['covariance_minimum_eigenvalue']>0 and
        r['variance_to_uniform']<=1+1e-8 for r in rows)
    result={'status':'passed' if passed else 'failed','maximum_scaled_KKT_violation':maximum,
        'rows':rows,'scope':'32 pre-2023 holdout covariances, numerical reference not future realized-risk optimum',
        'threshold_declared_before_audit':1e-4,'test2025_accessed':False}
    write_json(output,result);print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))
    if not passed:raise SystemExit(1)


if __name__=='__main__':main()
