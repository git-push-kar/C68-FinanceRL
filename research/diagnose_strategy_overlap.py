"""Training-period overlap among available causal financial strategies."""
import json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).parent
sys.path.insert(0,str(ROOT/'rl_readout_trial'))
from strategy_ppo import Market,NAMES
from metrics import metrics_from_infos

out=ROOT/'reports/strategy_overlap_training.json'
if out.exists():raise FileExistsError(out)
market=Market(str(ROOT/'sector_data_20261107/sector9_development_through2024.csv'),'2001-01-01','2018-12-31')
gross=[];net=[];rows=[];dates=None
for action,name in enumerate(NAMES):
    state=market.reset()
    while True:
        _,targets=state;state,_,done=market.step(targets[action])
        if done:break
    current=[r['return_date'] for r in market.infos]
    if dates is not None and dates!=current:raise AssertionError('Unpaired dates')
    dates=current;gross.append([r['gross_return'] for r in market.infos]);net.append([r['net_return'] for r in market.infos])
    rows.append({'strategy':name,'metrics':metrics_from_infos(market.infos)})
gross=np.asarray(gross);net=np.asarray(net);n=gross.shape[1]//5*5
weekly=np.prod(1+gross[:,:n].reshape(4,-1,5),axis=-1)-1
correlation=np.corrcoef(weekly)
result={'scope':'Training 2001-2018 only; descriptive action-space diagnostic, not a bound on achievable RL gains',
    'strategies':NAMES,'first_return':dates[0],'last_return':dates[-1],'sessions':len(dates),
    'complete_five_session_blocks':weekly.shape[1],
    'gross_five_session_return_correlation':correlation.tolist(),
    'minimum_pairwise_correlation':float(correlation[np.triu_indices(4,1)].min()),
    'annualized_daily_gross_volatility':(gross.std(1)*np.sqrt(252)).tolist(),
    'annualized_tracking_error_vs_equal_weight':((gross-gross[0]).std(1)*np.sqrt(252)).tolist(),
    'fixed_strategy_results':rows,'reserved_test_accessed':False,
    'interpretation_limit':'Highly correlated fixed strategies can still have exploitable relative patterns. Correlation alone does not establish unlearnability or model-size adequacy.'}
out.write_text(json.dumps(result,indent=2))
print(json.dumps({k:v for k,v in result.items() if k!='fixed_strategy_results'},indent=2))
