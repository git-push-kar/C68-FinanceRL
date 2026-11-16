"""Descriptive risk, costs and year slices from saved paths; never reselect models."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from return_comparison import statistics

ROOT=Path(__file__).parent


def describe(daily):
    r=np.array([d['net_return'] for d in daily]);turn=np.array([d['turnover'] for d in daily])
    base=statistics(r);downside=np.sqrt(np.mean(np.minimum(r,0)**2));tail=r[r<=np.quantile(r,.05)]
    return {**base,'sessions':len(r),'first':daily[0]['return_date'],'last':daily[-1]['return_date'],
        'annualized_volatility':float(np.sqrt(252)*r.std()),'arithmetic_annualized_mean':float(252*r.mean()),
        'Sortino_target_zero':float(np.sqrt(252)*r.mean()/downside) if downside else None,
        'Calmar':base['annualized_return']/base['max_drawdown'] if base['max_drawdown'] else None,
        'daily_historical_CVaR5':float(tail.mean()),'worst_daily_net_return':float(r.min()),
        'annualized_traded_stock_L1':float(252*turn.mean()),'annualized_half_L1_turnover':float(126*turn.mean()),
        'sum_cost_fractions':float(sum(d['trading_cost'] for d in daily))}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--prefix',required=True)
    p.add_argument('--output',required=True);args=p.parse_args();target=Path(args.output)
    if target.exists():raise FileExistsError(target)
    paths={}
    for seed in (42,123,456):
        folder=ROOT/'campaign_runs'/f'{args.prefix}{seed}';selection=json.loads((folder/'selection.json').read_text())
        paths[f'candidate_seed{seed}']=folder/f"development_{selection['selected']['step']}.json"
        paths[f'initial_seed{seed}']=folder/'initial_daily.json'
    for name in ('minimum_variance252_monthly','hrp252_monthly','equal_weight_daily'):
        paths[name]=ROOT/'campaign_runs/classical_development_cadence_verified_v2'/f'{name}.json'
    result={}
    for name,path in paths.items():
        daily=json.loads(path.read_text())['daily'];years=sorted({d['return_date'][:4] for d in daily})
        result[name]={'source':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'whole':describe(daily),'year_slices':{year:describe([d for d in daily if d['return_date'].startswith(year)]) for year in years}}
    payload={'role':'Descriptive reused-development paths; year slices retain executed positions/costs and are not fresh annual simulations.',
        'accounting':'Zero-lag historical development; final protocol separately uses a full close of delay. Risk-free/Sortino target zero; 252-session annualization.',
        'results':result,'test2025_accessed':False}
    target.parent.mkdir(parents=True,exist_ok=True);target.write_text(json.dumps(payload,indent=2,allow_nan=False))
    print(json.dumps({'output':str(target),'paths':len(paths)},indent=2))


if __name__=='__main__':main()
