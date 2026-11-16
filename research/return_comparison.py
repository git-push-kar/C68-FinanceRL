"""Paired, conditional return-series uncertainty and calendar breakdowns."""
import argparse
import json
from pathlib import Path
import numpy as np


def statistics(returns):
    r=np.asarray(returns,dtype=np.float64)
    equity=np.cumprod(1+r)
    peak=np.maximum.accumulate(np.r_[1.,equity])[1:]
    volatility=float(r.std())
    return {'return_total':float(equity[-1]-1),
        'annualized_return':float(equity[-1]**(252/len(r))-1),
        'sharpe_proxy':float(np.sqrt(252)*r.mean()/volatility) if volatility>0 else 0.,
        'max_drawdown':float(np.max(1-equity/peak)),'observations':len(r)}


def compare(a,b,reps=2000,block=20,seed=20261107):
    a,b=np.asarray(a),np.asarray(b)
    if len(a)!=len(b):raise ValueError('Unmatched return counts')
    rng=np.random.default_rng(seed);n=len(a);differences=[]
    for _ in range(reps):
        starts=rng.integers(0,n,size=int(np.ceil(n/block)))
        indices=np.concatenate([(start+np.arange(block))%n for start in starts])[:n]
        differences.append(statistics(a[indices])['sharpe_proxy']-statistics(b[indices])['sharpe_proxy'])
    return {'sharpe_difference':statistics(a)['sharpe_proxy']-statistics(b)['sharpe_proxy'],
        'paired_circular_block_bootstrap_95pct':np.quantile(differences,[.025,.975]).tolist(),
        'replicates':reps,'block_length':block,'seed':seed,
        'note':'Conditional on these generated return series. Not independent markets, retraining uncertainty or a fresh-test significance claim.'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--candidate',required=True);p.add_argument('--reference',required=True)
    p.add_argument('--output',required=True)
    args=p.parse_args();output=Path(args.output)
    if output.exists():raise FileExistsError(output)
    a=json.loads(Path(args.candidate).read_text())['daily'];b=json.loads(Path(args.reference).read_text())['daily']
    dates=[r['return_date'] for r in a]
    if dates!=[r['return_date'] for r in b]:raise ValueError('Return dates do not align')
    x=np.array([r['net_return'] for r in a]);y=np.array([r['net_return'] for r in b])
    result={'candidate':args.candidate,'reference':args.reference,'paired_comparison':compare(x,y),
        'candidate_metrics':statistics(x),'reference_metrics':statistics(y),'years':{}}
    for year in sorted(set(d[:4] for d in dates)):
        mask=np.array([d.startswith(year) for d in dates])
        result['years'][year]={'candidate':statistics(x[mask]),'reference':statistics(y[mask])}
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2,allow_nan=False))
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
