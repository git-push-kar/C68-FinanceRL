"""CPU-only diagnostic: dynamic policy versus its validation-average mixture.

The constant mixture is derived only from already-recorded validation actions,
never from development returns. This is an explanatory control, not a fresh
test or a replacement for a separately trained allocation baseline.
"""
import argparse,json,sys
from pathlib import Path
import numpy as np

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True)
    p.add_argument('--branch',choices=['rl_strategy_trial','rl_sector_trial','rl_text_trial','rl_counterfactual_trial','rl_readout_trial','rl_summary_numeric_control'],required=True)
    p.add_argument('--step',type=int);a=p.parse_args()
    root=Path(__file__).parent;run=Path(a.run).resolve()
    sys.path.insert(0,str(root/a.branch))
    from strategy_ppo import Market,NAMES
    from metrics import metrics_from_infos
    if a.step is None:
        step=json.loads((run/'selection.json').read_text())['selected']['step']
    else:step=a.step
    val=json.loads((run/f'validation_{step}.json').read_text())
    dev=json.loads((run/f'development_{step}.json').read_text())
    ps=np.array(val['probabilities']);mix=ps.mean(0);mix/=mix.sum()
    if a.branch=='rl_strategy_trial':
        data=root/'data/nifty50_prices.csv';start='2023-01-01'
    else:
        data=root/'sector_data_20261107/sector9_development_through2024.csv';start='2022-01-01'
    market=Market(str(data),start,'2024-12-31');state=market.reset()
    while True:
        _,targets=state;state,_,done=market.step(mix@targets)
        if done:break
    fixed=metrics_from_infos(market.infos)
    if [d['return_date'] for d in dev['daily']] != [d['return_date'] for d in market.infos]:
        raise ValueError('Comparison dates differ')
    dp=np.array(dev['probabilities'])
    result={'scope':'Reused development diagnostic; constant mixture fitted to validation probabilities only',
        'step':step,'checkpoint_final_selection':a.step is None,'strategies':NAMES,
        'validation_average_mixture':mix.tolist(),'validation_probability_std':ps.std(0).tolist(),
        'development_probability_std':dp.std(0).tolist(),
        'dynamic_metrics':dev['metrics'],'constant_mixture_metrics':fixed,
        'dynamic_minus_constant_Sharpe':dev['metrics']['sharpe_proxy']-fixed['sharpe_proxy'],
        'interpretation_limit':'This compares implemented policies, not a causal attribution to timing. Portfolio histories and costs differ. No fresh test accessed.'}
    out=run/f'adaptation_diagnostic_{step}.json'
    if out.exists():raise FileExistsError(out)
    out.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

if __name__=='__main__':main()
