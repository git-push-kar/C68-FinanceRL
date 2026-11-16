"""Additional risk/cost measures for a saved, actually simulated return path."""
import argparse
import json
from pathlib import Path
import numpy as np
from return_comparison import statistics


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',required=True);p.add_argument('--output',required=True)
    args=p.parse_args();output=Path(args.output)
    if output.exists():raise FileExistsError(output)
    daily=json.loads(Path(args.input).read_text())['daily']
    returns=np.array([r['net_return'] for r in daily]);turnover=np.array([r['turnover'] for r in daily])
    base=statistics(returns);downside=float(np.sqrt(np.mean(np.minimum(returns,0)**2)))
    tail=returns[returns<=np.quantile(returns,.05)]
    result={**base,'arithmetic_annualized_mean_return':float(252*returns.mean()),
        'annualized_volatility':float(np.sqrt(252)*returns.std()),
        'sortino_target_zero':float(np.sqrt(252)*returns.mean()/downside) if downside else None,
        'calmar_CAGR_over_max_drawdown':base['annualized_return']/base['max_drawdown'] if base['max_drawdown'] else None,
        'daily_historical_CVaR_5pct':float(tail.mean()),'annualized_traded_stock_weight_L1':float(252*turnover.mean()),
        'annualized_half_L1_turnover':float(126*turnover.mean()),
        'sum_cost_fractions':float(sum(r['trading_cost'] for r in daily)),
        'mean_daily_cost_fraction':float(np.mean([r['trading_cost'] for r in daily])),
        'conventions':'Primary accounting charges 10 bps per unit of traded-stock L1 weight, including cash entry; half-L1 is provided for terminology comparison. CAGR and arithmetic annual mean are separate.',
        'role':'descriptive development risk metrics; no significance or new-test claim'}
    output.write_text(json.dumps(result,indent=2,allow_nan=False));print(json.dumps(result,indent=2))


if __name__=='__main__':main()
