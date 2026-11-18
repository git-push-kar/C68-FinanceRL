"""Pure-strategy and fixed-mixture controls for a completed native selector."""
import argparse,json
from pathlib import Path
import numpy as np
from candidate_market import Market,evaluate,NAMES
from model import write_json

p=argparse.ArgumentParser();p.add_argument('--run',required=True);r=Path(p.parse_args().run)
s=json.loads((r/'selection.json').read_text())['selected']
val=json.loads((r/f'validation_{s["step"]}.json').read_text())
dev=json.loads((r/f'development_{s["step"]}.json').read_text())
matrix=np.asarray(val['probabilities'])
if np.max(abs(matrix-matrix[0]))>1e-12:raise AssertionError('Expected a constant strategy mixture')
mix=matrix.mean(0);mix/=mix.sum()
m=Market('../multiasset_data_20261115/five_asset_development_through2024.csv','2022-01-01','2024-12-31')
controls={name:evaluate(None,m,'cpu',fixed=i) for i,name in enumerate(NAMES)}
controls['validation_average_mixture']=evaluate(None,m,'cpu',fixed=mix)
write_json(r/'candidate_controls.json',{'scope':'Reused development; optimizer alone is an explicit comparator, not adapter learning.',
    'validation_average_mixture':mix.tolist(),'controls':controls,
    'dynamic_minus_constant_Sharpe':dev['metrics']['sharpe_proxy']-controls['validation_average_mixture']['metrics']['sharpe_proxy']})
