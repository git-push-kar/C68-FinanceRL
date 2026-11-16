"""All declared CPU walk-forward folds, without selecting a winning period."""
import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).parent
lines=['# Full-history direct RL: historical CPU walk-forward diagnostic','',
    '**Reused development history; not native InternVL results or a prospective untouched test.**','',
    '| Development period | Mean paired Sharpe change | Positive seeds | 20-day joint seed/time interval |',
    '|---|---:|---:|---|']
means=[];records=[]
for fold,period in [(1,'2013-2015'),(2,'2016-2018'),(3,'2019-2021'),(4,'2022-2024')]:
    name=f'exposure_walkforward_f{fold}_seed_comparison_budget64.json' if fold<4 else 'exposure_global_numeric_seed_comparison_budget64.json'
    result=json.loads((ROOT/'reports'/name).read_text());ci=result['paired_95_intervals']['20']['shared_time_blocks_and_resampled_seeds']
    means.append(result['mean_Sharpe_gain']);records.append({'fold':fold,'period':period,'report':name,'result':result})
    lines.append(f'| {period} | {result["mean_Sharpe_gain"]:+.6f} | {result["positive_seeds"]}/3 | [{ci[0]:+.6f}, {ci[1]:+.6f}] |')
lines+=['',f'Equal-weight mean of the four paired fold means: {np.mean(means):+.6f}. '
    'This is not a pooled Sharpe or a stitched investable portfolio. Each fold resets to cash; '
    'boundary liquidation costs have not been modelled. No best-fold-only claim is made.','',
    'Each checkpoint is selected solely on its immediately preceding validation period. '
    'The method was designed after observing later development outcomes, so these historical folds '
    'do not remove adaptive method-selection bias. All seed records, reload checks and fixed/causal '
    'controls are retained in their respective run folders.','']
(ROOT/'reports/EXPOSURE_WALKFORWARD.md').write_text('\n'.join(lines),encoding='utf-8')
(ROOT/'reports/exposure_walkforward_summary.json').write_text(json.dumps({'scope':'Reused historical CPU diagnostic','equal_weight_mean_fold_Sharpe_gain':float(np.mean(means)),'folds':records},indent=2))
print(ROOT/'reports/EXPOSURE_WALKFORWARD.md')
