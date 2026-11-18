"""Training-only descriptive Q-value diagnostic; no validation/test fitting."""
import json,os,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).parent;sys.path.insert(0,str(ROOT/'rl_exposure_trial'))
from exposure_market import Market,LEVELS
from counterfactual import action_values,choose
market=Market(ROOT/'sector_data_20261107/sector9_development_through2024.csv','2001-01-01','2018-12-31')
rng=np.random.default_rng(20261108);rows=[]
for _ in range(512):
    start=int(rng.integers(market.env.start_index,len(market.env.prices)-61));state=market.reset(start)
    for _ in range(4):state,_,_=market.step(state[1][choose(np.ones(4)/4,rng.random())])
    obs,targets=state;values=action_values(market,targets,lambda x:np.ones(4)/4,rng.random(2))
    rows.append({'lagged_annual_volatility_percent':float(obs['summary'][4]),'values':values.tolist()})
vol=np.array([x['lagged_annual_volatility_percent'] for x in rows]);q=np.array([x['values'] for x in rows]);q-=q.mean(1,keepdims=True)
edges=np.quantile(vol,[0,.2,.4,.6,.8,1]);bins=[]
for i in range(5):
    mask=(vol>=edges[i]) & ((vol<edges[i+1]) if i<4 else (vol<=edges[i+1]))
    means=q[mask].mean(0)
    bins.append({'lagged_volatility_range_percent':[float(edges[i]),float(edges[i+1])],
        'sample_count':int(mask.sum()),'mean_centered_MC_utility':means.tolist(),
        'highest_sample_mean_exposure':float(LEVELS[np.argmax(means)])})
result={'scope':'Training 2001-2018 only, descriptive diagnostic; no fitted trading policy or held-out performance',
    'states':512,'seed':20261108,'continuation':'Uniform categorical policy, shared continuation uniforms across four actions, 3-decision horizon',
    'reward':'100*(daily net return - 1.5*daily net return squared)',
    'volatility_bins':bins,'limitations':'Noisy MC estimates, shared/overlapping historical samples; conditional sample means do not prove exploitable out-of-sample differences.',
    'reserved_test_accessed':False}
out=ROOT/'reports/exposure_training_signal.json'
if out.exists():raise FileExistsError(out)
out.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
