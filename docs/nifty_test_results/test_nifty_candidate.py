import os,sys,json,hashlib
from pathlib import Path
import numpy as np
R=Path.cwd();O=R/'reports/nifty_final_candidate_test_20261116';O.mkdir(exist_ok=False);sys.path.insert(0,str(R/'rl_candidate_static_native'));os.environ['CUDA_VISIBLE_DEVICES']='-1'
import execution_policy
execution_policy.wait_for_compute_window=lambda:None
import candidate_market,exposure_market,cash_execution
from metrics import metrics_from_infos
import torch
torch.set_num_threads(2)
reload=json.loads((R/'reports/profit_improving_runs_audit_20261116/actual_weight_reload.json').read_text());entries=reload['entries'];assert max(e['max_probability_error'] for e in entries)<1e-6
p0=np.array(entries[0]['probabilities']);assert all(np.max(abs(np.array(e['disabled_adapter_probabilities'])-p0))<1e-6 for e in entries)
policies={'untrained':p0};policies.update({e['run'].replace('nifty_static_native_',''):np.array(e['probabilities']) for e in entries if e['step']>0})
protocol={'training':'2019-2021','validation':'2022','excluded_main_results':'2023-2024: used for method development; retained only in historical appendix','test_intervals':{'test_2025':['2025-01-01','2025-12-31'],'test_jan2026':['2026-01-01','2026-01-30'],'test_combined':['2025-01-01','2026-01-30']},'status':'Retrospective test of a candidate revisited after prior results; data previously accessed. Not untouched confirmatory evidence. All intervals and cost levels reported, no checkpoint selection. Combined test carries holdings continuously; separate intervals start independently from cash.','cost_bps':[10,20,50],'buy_hold':'Equal initial weights, plus hold-first-trained-target and hold-first-untrained-target controls; each initial purchase fee charged; no terminal liquidation','bootstrap':'Paired circular blocks of5/20/60 sessions,5000 replicates; CI conditional on selected candidate, not adjusted for historical search; three seeds do not create three independent markets','policies':{k:v.tolist() for k,v in policies.items()}}
(O/'protocol.json').write_text(json.dumps(protocol,indent=2));shutil=None
rows=[];stored={}
for interval,(start,end) in protocol['test_intervals'].items():
 for cost in protocol['cost_bps']:
  def execute(*args,**kwargs):kwargs['fee_bps']=cost;return cash_execution.execute(*args,**kwargs)
  exposure_market.execute=execute;candidate_market.execute=execute
  market=candidate_market.Market(str(R/'data/nifty50_prices.csv'),start,end);targets=market.reset()[1];e=market.env
  for name,weights in policies.items():
   result=candidate_market.evaluate(None,market,'cpu',fixed=weights.copy());stored[(interval,cost,name)]=result
  for i,name in enumerate(['equal_weight_rebalanced','inverse_vol_rebalanced','momentum_rebalanced','reversal_rebalanced']):stored[(interval,cost,name)]=candidate_market.evaluate(None,market,'cpu',fixed=i)
  holdings={'buy_hold_equal':np.ones(e.n_assets)/e.n_assets,'buy_hold_trained_initial':policies['seed42_64']@targets,'buy_hold_untrained_initial':policies['untrained']@targets}
  for name,weights in holdings.items():
   i=e.start_index;fee=100000*(cost/10000)/(1+cost/10000);units=(100000-fee)*weights/e.prices[i];prev=peak=100000.;daily=[]
   for j in range(i+1,len(e.prices)):
    eq=float(units@e.prices[j]);net=eq/prev-1;peak=max(peak,eq);gross=float((units@e.prices[j])/(units@e.prices[j-1])-1)
    daily.append({'net_return':net,'gross_return':gross,'trading_cost':fee/100000 if j==i+1 else 0.,'turnover':(100000-fee)/100000 if j==i+1 else 0.,'equity':eq,'drawdown':1-eq/peak,'decision_date':str(e.dates[j-1])[:10],'return_date':str(e.dates[j])[:10]});prev=eq
   stored[(interval,cost,name)]={'metrics':metrics_from_infos(daily),'daily':daily}
  for (period,c,n),result in list(stored.items()):
   if period==interval and c==cost:
    row={'interval':period,'fee_bps':c,'policy':n,**result['metrics']};rows.append(row);(O/f'{period}_{c}bp_{n}.json').write_text(json.dumps(result,indent=2))
  print(interval,cost,[(r['policy'],round(r['return_total']*100,3)) for r in rows if r['interval']==interval and r['fee_bps']==cost],flush=True)
(O/'metrics.json').write_text(json.dumps(rows,indent=2));bootstrap=[];rng=np.random.default_rng(20261116)
for interval in protocol['test_intervals']:
 train=np.array([[d['net_return'] for d in stored[(interval,10,name)]['daily']] for name in policies if name!='untrained']);n=train.shape[1]
 for baseline in ['untrained','buy_hold_equal','buy_hold_trained_initial','inverse_vol_rebalanced']:
  base=np.array([d['net_return'] for d in stored[(interval,10,baseline)]['daily']]);assert len(base)==n
  for block in [5,20,60]:
   start=rng.integers(n,size=(5000,int(np.ceil(n/block))));idx=((start[:,:,None]+np.arange(block))%n).reshape(5000,-1)[:,:n];bt=train[:,idx];bb=base[idx]
   diff=(np.prod(1+bt,axis=2)-np.prod(1+bb,axis=1)[None,:]).mean(0)*100
   sharp=lambda x:np.mean(x,axis=-1)/np.std(x,axis=-1)*np.sqrt(252)
   sd=(sharp(bt)-sharp(bb)[None,:]).mean(0)
   bootstrap.append({'interval':interval,'baseline':baseline,'block':block,'return_difference_pp_95CI':np.quantile(diff,[.025,.975]).tolist(),'sharpe_difference_95CI':np.quantile(sd,[.025,.975]).tolist()})
(O/'paired_intervals.json').write_text(json.dumps(bootstrap,indent=2));print('FINISHED',flush=True)
