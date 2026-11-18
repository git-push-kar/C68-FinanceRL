import argparse,json
from pathlib import Path
import numpy as np
import torch
from candidate_market import Market,evaluate,NAMES
from numeric_policy import NumericPolicy
from model import write_json

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True);a=parser.parse_args();run=Path(a.run)
    torch.set_num_threads(2)
    selected=json.loads((run/'selection.json').read_text())['selected']
    val=json.loads((run/f'validation_{selected["step"]}.json').read_text())
    mixture=np.asarray(val['probabilities']).mean(0);mixture/=mixture.sum()
    dev=Market('../data/nifty50_prices.csv','2023-01-01','2024-12-31')
    policy=NumericPolicy(dev.env.tickers,selected['package'])
    controls={name:evaluate(policy,dev,'cpu',fixed=i) for i,name in enumerate(NAMES)}
    controls['validation_average_mixture']=evaluate(policy,dev,'cpu',fixed=mixture)
    trained=json.loads((run/f'development_{selected["step"]}.json').read_text())
    write_json(run/'candidate_controls.json',{'scope':'Reused development, exact fee ledger; controls do not select checkpoint',
        'validation_average_mixture':mixture.tolist(),'controls':controls,
        'dynamic_minus_constant_Sharpe':trained['metrics']['sharpe_proxy']-controls['validation_average_mixture']['metrics']['sharpe_proxy']})

if __name__=='__main__':main()
