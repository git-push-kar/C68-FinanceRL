"""Paired return uncertainty and state/action audit of completed run3 artifacts.

Run only after the training worker finishes. Resampling is descriptive under
adaptive retrospective selection, not independent confirmation of an edge.
"""
import argparse,json
from pathlib import Path
import numpy as np

def paired_interval(a,b,capital,block,seed=42,repeats=2000):
 a=np.asarray(a,float);b=np.asarray(b,float)
 if len(a)!=len(b) or not len(a):raise ValueError('Mismatched observation periods')
 ra=np.diff(np.log(np.r_[capital,a]));rb=np.diff(np.log(np.r_[capital,b]));n=len(a)
 rng=np.random.default_rng(seed);values=[]
 for _ in range(repeats):
  starts=rng.integers(0,n,int(np.ceil(n/block)));indices=((starts[:,None]+np.arange(block))%n).ravel()[:n]
  values.append(float(np.expm1(ra[indices].sum())-np.expm1(rb[indices].sum())))
 return {'block_sessions':block,'replicates':repeats,'return_difference':float(a[-1]/capital-b[-1]/capital),'descriptive_95pct_interval':np.quantile(values,[.025,.975]).tolist()}

def summarize(run):
 run=Path(run);status=json.loads((run/'status.json').read_text())
 if status['state']!='completed':raise RuntimeError('Worker is not complete; no partial headline')
 metrics=json.loads((run/'test_metrics.json').read_text());results=[]
 for r in metrics:
  if r['policy']!='trained':continue
  capital=r['capital'];prefix=f"test_{r['interval']}_{int(capital)}_";trained=json.loads((run/(prefix+'trained.json')).read_text())
  comparison={}
  for other in ['untrained','supervised_only','anchor','buy_hold','equal_rebalanced']:
   baseline=json.loads((run/(prefix+other+'.json')).read_text())
   comparison[other]=[paired_interval(trained['daily_equity'],baseline['daily_equity'],capital,block) for block in [20,60]]
  decisions=trained['decisions'];total_bought=sum(sum(max(q,0) for q in d['trade_shares']) for d in decisions);total_sold=sum(sum(max(-q,0) for q in d['trade_shares']) for d in decisions)
  results.append({'interval':r['interval'],'capital':capital,'metrics':r,'comparisons':comparison,'decision_count':len(decisions),'real_order_dates':sum(any(q for q in d['trade_shares']) for d in decisions),'shares_bought':total_bought,'shares_sold':total_sold,'mean_signal_stock_exposure':float(np.mean([sum(s['weight'] for s in d['observation']['stocks']) for d in decisions]))})
 capital_comparisons=[]
 primary=[r for r in results if r['interval']=='primary_continuous']
 for a,b in zip(primary,primary[1:]):
  x=json.loads((run/f"test_primary_continuous_{int(a['capital'])}_trained.json").read_text());y=json.loads((run/f"test_primary_continuous_{int(b['capital'])}_trained.json").read_text())
  capital_comparisons.append({'capital_a':a['capital'],'capital_b':b['capital'],'different_action_dates':sum(d['actions']!=e['actions'] for d,e in zip(x['decisions'],y['decisions'])),'different_executed_share_dates':sum(d['trade_shares']!=e['trade_shares'] for d,e in zip(x['decisions'],y['decisions'])),'return_difference':y['metrics']['return']-x['metrics']['return']})
 output={'selected':status['selected'],'caveat':'Circular paired block-bootstrap on one adaptively reused retrospective sample; not independent confirmation, not corrected for experiment selection. Similar high-capital returns are not failure. Different share quantities alone do not prove learned budget preferences. Exposure is observed at decision dates, not daily average.','results':results,'capital_comparisons':capital_comparisons}
 (run/'descriptive_uncertainty_and_action_audit.json').write_text(json.dumps(output,indent=2));return output

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('run');a=p.parse_args();print(json.dumps(summarize(a.run),indent=2))
