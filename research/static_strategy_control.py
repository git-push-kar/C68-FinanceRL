"""Validation-selected constant mixture; no backbone and no reserved data."""
import argparse,itertools,json,sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).parent

def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',choices=['nifty','sector'],required=True)
    a=p.parse_args();out=ROOT/'reports'/f'{a.dataset}_validation_static_grid.json'
    if out.exists():raise FileExistsError(out)
    branch='rl_strategy_trial' if a.dataset=='nifty' else 'rl_sector_trial'
    sys.path.insert(0,str(ROOT/branch))
    from strategy_ppo import Market,NAMES
    from metrics import metrics_from_infos
    from model import digest
    if a.dataset=='nifty':
        data=ROOT/'data/nifty50_prices.csv';val_dates=('2022-01-01','2022-12-31');dev_dates=('2023-01-01','2024-12-31')
    else:
        data=ROOT/'sector_data_20261107/sector9_development_through2024.csv';val_dates=('2019-01-01','2021-12-31');dev_dates=('2022-01-01','2024-12-31')
    val=Market(str(data),*val_dates);dev=Market(str(data),*dev_dates)
    def evaluate(market,weights):
        state=market.reset()
        while True:
            _,targets=state;state,_,done=market.step(np.asarray(weights,dtype=np.float32)@targets)
            if done:break
        return metrics_from_infos(market.infos)
    candidates=[]
    # Fixed coarse simplex grid: 35 candidates, all declared before outcomes.
    for counts in itertools.product(range(5),repeat=4):
        if sum(counts)!=4:continue
        weights=(np.asarray(counts)/4).tolist()
        candidates.append({'weights':weights,'validation':evaluate(val,weights)})
    selected=max(candidates,key=lambda c:c['validation']['sharpe_proxy'])
    result={'scope':'Reused development only; conventional static control, not RL',
        'selection':'35 fixed quarter-increment simplex candidates; highest validation net Sharpe; lexicographic first tie',
        'actions':NAMES,'validation_dates':val_dates,'development_dates':dev_dates,
        'dataset_sha256':digest(data),'implementation_sha256':digest(__file__),
        'selected':selected,'development':evaluate(dev,selected['weights']),
        'candidates':candidates,'reserved_test_accessed':False,
        'limitations':'Validation search has its own selection bias. Fixed strategy weights still produce market-dependent stock targets. This controls the value of switching between strategies, not all market adaptation.'}
    out.write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='candidates'},indent=2))

if __name__=='__main__':main()
