"""Paired attribution of trainable LoRA within otherwise matched finance PPO."""
import json
from pathlib import Path
import numpy as np
from compare_rl_seeds import sharpe
ROOT=Path(__file__).parent
def read(p):return json.loads(p.read_text())
def main():
 out=ROOT/'reports/full_frozen_ppo_paired128.json'
 if out.exists():raise FileExistsError(out)
 rows=[];full=[];frozen=[];dates=None
 for seed in (42,123,456):
  suffix='256_persistent' if seed==42 else '128'
  names=[f'strategy_ppo_internvl_seed{seed}_{suffix}',f'strategy_ppo_no_lora_seed{seed}_128']
  pair=[];details=[]
  for name in names:
   run=ROOT/'campaign_runs'/name
   if read(run/'status.json')['status']!='completed':raise AssertionError('Unfinished run')
   cs=[c for c in read(run/'candidates.json') if c['step'] in (0,32,64,96,128)]
   if {c['step'] for c in cs}!={0,32,64,96,128}:raise AssertionError('Incomplete common selection grid')
   chosen=max(cs,key=lambda c:(c['validation']['sharpe_proxy'],-c['step']))
   d=read(run/f'development_{chosen["step"]}.json');base=read(run/'development_0.json')
   ds=[v['return_date'] for v in d['daily']]
   if dates is not None and ds!=dates:raise AssertionError('Unpaired market dates')
   dates=ds;pair.append([v['net_return'] for v in d['daily']])
   details.append({'run':name,'selected_step':chosen['step'],'initial':base['metrics'],'selected':d['metrics'],
    'gain_over_initial':d['metrics']['sharpe_proxy']-base['metrics']['sharpe_proxy']})
  full.append(pair[0]);frozen.append(pair[1]);rows.append({'seed':seed,'full':details[0],'frozen':details[1]})
 x=np.asarray(frozen);y=np.asarray(full);gains=sharpe(y)-sharpe(x);n=x.shape[1];rng=np.random.default_rng(20261108);intervals={}
 for block in (5,20,60):
  temporal=[];joint=[]
  for _ in range(5000):
   starts=rng.integers(0,n,size=(n+block-1)//block);idx=((starts[:,None]+np.arange(block))%n).ravel()[:n]
   d=sharpe(y[:,idx])-sharpe(x[:,idx]);temporal.append(d.mean());joint.append(d[rng.integers(0,3,size=3)].mean())
  intervals[str(block)]={'paired_time':np.quantile(temporal,[.025,.975]).tolist(),'paired_time_and_seed':np.quantile(joint,[.025,.975]).tolist()}
 result={'scope':'Reused development; adaptive method search unadjusted; all three seeds',
  'budget':128,'selection_grid':[0,32,64,96,128],'selection':'2022 validation net Sharpe, earliest tie',
  'individual_full_minus_frozen_Sharpe':gains.tolist(),'mean_full_minus_frozen_Sharpe':float(gains.mean()),
  'paired_95_intervals':intervals,'runs':rows,
  'interpretation':'This estimates the incremental effect of allowing LoRA training alongside learned encoders and policy heads. Whole-policy improvement against initialization is a separate claim.'}
 out.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
