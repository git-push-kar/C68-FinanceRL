"""Causal, auditable curriculum; never used as an inference fallback."""
import json
from pathlib import Path
import numpy as np
from market import Market,evaluate,LEVELS

KINDS=['inverse_vol','trend_tilt','absolute_trend']

def anchor(obs,kind):
 rows=obs['stocks'];vol=np.maximum([r['vol20'] for r in rows],.10)
 trend=np.array([.5*r['return60']+.5*r['return120'] for r in rows])
 affordable=np.array([r['price']*1.0015005<=obs['equity'] for r in rows])
 score=1/vol
 if kind=='trend_tilt':score*=np.exp(np.clip(trend/vol,-2,2))
 if kind=='absolute_trend':
  active=(trend>0)&affordable
  order=np.argsort(np.where(active,trend/vol,-np.inf));active&=np.isin(np.arange(5),order[-3:])
  score*=active
 score*=affordable
 weights=score/max(score.sum(),1e-12)
 # Quantization uses the same advertised action vocabulary as the model.
 levels=np.array([0,.05,.10,.20,.30,obs['max_stock_weight']])
 actions=np.abs(weights[:,None]-levels[None,:]).argmin(1)+1
 # A turnover deadband: KEEP is legitimate only when close to the desired weight.
 current=np.array([r['weight'] for r in rows]);actions=np.where((abs(current-weights)<.025)&(current>0),0,actions)
 return actions.astype(int)

def audit(out):
 m=Market();rows=[]
 for kind in KINDS:
  for year in range(2015,2025):
   result=evaluate(m,lambda o:(anchor(o,kind),{'anchor':kind}),f'{year}-01-01',f'{year}-12-31',100000.)
   rows.append({'kind':kind,'year':year,**result['metrics']})
 scores={k:float(np.mean([np.log1p(r['return'])-.1*r['max_drawdown'] for r in rows if r['kind']==k and r['year']<=2023])) for k in KINDS}
 selected=max(KINDS,key=lambda k:scores[k]);result={'selected':selected,'criterion':'Mean annual net log growth minus 0.1 drawdown, 2015-2023 only; 2024 not used for anchor selection','scores':scores,'rows':rows}
 Path(out).write_text(json.dumps(result,indent=2));return result

if __name__=='__main__':
 import sys
 result=audit(sys.argv[1]);print(json.dumps(result,indent=2))
