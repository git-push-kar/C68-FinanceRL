"""CPU signal diagnostic, not an InternVL adapter or substitute final result.

Expanding-window ridge forecasts with purged20-session labels. Evaluate2022,
2023,2024 only; do not inspect2025-2026 to select a configuration.
"""
import sys,json,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'stateful_rl_20261117_run4'))
from market import Market,evaluate,baseline

def features(o):
 rows=o['stocks'];keys=['return5','return20','return60','return120','vol20','drawdown60','volume_ratio']
 x=np.array([[r[k] for k in keys] for r in rows]);x[:,-1]=np.log(np.maximum(x[:,-1],.01))
 market=x.mean(0);relative=x-market;vol=np.maximum(x[:,4:5],.10)
 return np.column_stack([x,relative,np.broadcast_to(market,x.shape),x[:,:4]/vol])

def fit(x,y,alpha):
 mean=x.mean(0);std=np.maximum(x.std(0),1e-5);z=(x-mean)/std;target_mean=y.mean();coef=np.linalg.solve(z.T@z+alpha*np.eye(z.shape[1]),z.T@(y-target_mean))
 return lambda obs:((features(obs)-mean)/std)@coef+target_mean

def choose(predict):
 def policy(obs):
  prediction=predict(obs);selected=np.argsort(prediction)[-3:];weights=np.zeros(5)
  for i in selected:
   if prediction[i]>0:weights[i]=1/3
  return weights
 return policy

def main(out):
 m=Market();first,_=m.bounds('2015-01-01','2024-12-31');rows=[];folds=[]
 for year in [2022,2023,2024]:
  _,cutoff=m.bounds('2015-01-01',f'{year-1}-12-31');x=[];y=[];dates=[]
  for t in range(first+1,cutoff-20+1,5):
   obs=m.observe(t,100000.,np.zeros(5,int),100000.);x.extend(features(obs));dates.append(str(m.dates[t]))
   # Buy next open, hold20sessions, sell at final close; economic total returns.
   gross=(m.index[t+20]/m.index[t+1])*(m.close[t+1]/m.open[t+1]);y.extend(np.log(gross)-.003)
  x=np.asarray(x);y=np.asarray(y);folds.append({'year':year,'training_decision_last':dates[-1],'latest_reward_date':str(m.dates[cutoff]),'training_stock_examples':len(y)})
  for alpha in [10.,100.,1000.]:
   predictor=fit(x,y,alpha);result=evaluate(m,choose(predictor),f'{year}-01-01',f'{year}-12-31',100000.)
   rows.append({'year':year,'method':'pooled_ridge','alpha':alpha,**result['metrics']})
  for name in ['buy_hold','equal_rebalanced','inverse_vol','momentum','cash']:
   result=evaluate(m,baseline(name),f'{year}-01-01',f'{year}-12-31',100000.);rows.append({'year':year,'method':name,**result['metrics']})
 result={'scope':'CPU diagnostic only. Asset-shared linear numerical forecasting, no ticker identity, learned training-only standardization, purged forward labels. Does not demonstrate trained InternVL/RL performance. Same execution engine and costs. No test-period search.','folds':folds,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'results':rows}
 Path(out).write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

if __name__=='__main__':main(sys.argv[1])
