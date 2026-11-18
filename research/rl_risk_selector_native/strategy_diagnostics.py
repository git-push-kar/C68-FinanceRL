"""Historical regime diagnostics; excludes the new reserved data entirely."""
import json
from pathlib import Path
import numpy as np
from strategy_ppo import Market,evaluate,NAMES

out=Path('../reports/strategy_regime_diagnostics.json')
if out.exists():raise FileExistsError(out)
result={}
for name,start,end in [('train','2019-01-01','2021-12-31'),('selection','2022-01-01','2022-12-31'),
                       ('development_2023','2023-01-01','2023-12-31'),('development_2024','2024-01-01','2024-12-31')]:
    market=Market('../data/nifty50_prices.csv',start,end)
    result[name]={}
    for i,strategy in enumerate(NAMES):result[name][strategy]=evaluate(None,market,'cpu',i)['metrics']
out.write_text(json.dumps({'role':'Post-pilot historical diagnostic, not new test results','results':result},indent=2))
print(json.dumps({k:{n:round(m['sharpe_proxy'],4) for n,m in v.items()} for k,v in result.items()},indent=2))
