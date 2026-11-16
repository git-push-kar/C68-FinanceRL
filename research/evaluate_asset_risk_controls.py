"""Causal traditional references inside the learned five-asset weight bounds."""
import os,sys,json
from pathlib import Path
R=Path(__file__).parent;os.environ['CUDA_VISIBLE_DEVICES']='-1';sys.path.insert(0,str(R/'rl_asset_allocation_dynamic'));os.chdir(R/'rl_asset_allocation_dynamic')
import numpy as np
from scipy.optimize import minimize
from candidate_market import Market
from metrics import metrics_from_infos
out=R/'reports/asset_allocation_risk_controls.json'
if out.exists():raise FileExistsError(out)
results={}
for label,start,end in [('train','2010-01-01','2018-12-31'),('validation','2019-01-01','2021-12-31'),('development','2022-01-01','2024-12-31')]:
 results[label]={}
 for method in ('equal_weight','inverse_volatility126','minimum_variance126_shrink10'):
  m=Market('../multiasset_data_20261115/five_asset_development_through2024.csv',start,end);state=m.reset();previous=np.ones(5)/5
  while True:
   e=m.env;i=e.index-1;history=e.prices[i-125:i+1]/e.prices[i-126:i]-1
   if method=='equal_weight':target=np.ones(5)/5
   elif method=='inverse_volatility126':
    inv=1/history.std(0).clip(1e-5);inv/=inv.sum();target=.105+.475*inv
   else:
    cov=np.cov(history,rowvar=False,bias=True)*252;cov=.9*cov+.1*np.diag(np.diag(cov))
    fit=minimize(lambda w:float(w@cov@w),previous,jac=lambda w:2*cov@w,bounds=[(.105,.58)]*5,constraints=[{'type':'eq','fun':lambda w:w.sum()-1,'jac':lambda w:np.ones(5)}],method='SLSQP',options={'ftol':1e-12,'maxiter':200})
    if not fit.success or abs(fit.x.sum()-1)>1e-10:raise RuntimeError('Risk optimizer failed')
    target=fit.x/fit.x.sum()
   if target.min()<.105-1e-10 or target.max()>.58+1e-10:raise AssertionError('Bounds differ')
   previous=target;state,_,done=m.step(target)
   if done:break
  results[label][method]={'metrics':metrics_from_infos(m.infos),'daily':m.infos};print(label,method,results[label][method]['metrics']['sharpe_proxy'],flush=True)
out.write_text(json.dumps({'scope':'Fixed causal references; no policy selection, reserved outcomes or fitted forward-return targets','bounds':[.105,.58],'fees_bps_full_turnover':10,'decision_interval':5,'signal_lag':1,'results':results},indent=2))
