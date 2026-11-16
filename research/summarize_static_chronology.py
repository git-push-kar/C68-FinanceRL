import json
from pathlib import Path
import numpy as np
R=Path(__file__).parent;rows=[]
for year,file in [('2021','static_chronology_f1.json'),('2022','static_chronology_f2.json'),('2023-2024','static_native_seed_comparison_budget64.json')]:
 d=json.loads((R/'reports'/file).read_text());row={'period':year,'positive_seeds':d['positive_seeds'],'mean_Sharpe_gain':d['mean_Sharpe_gain'],'interval20':d['paired_95_intervals']['20']['shared_time_blocks_and_resampled_seeds']}
 for key in ('sharpe_proxy','return_total','max_drawdown'):
  row[key]={'initial':float(np.mean([r['initial'][key] for r in d['runs']])),'trained':float(np.mean([r['trained'][key] for r in d['runs']]))}
 rows.append(row)
result={'scope':'Historical method development; periods not untouched and training histories overlap. Separate fold portfolios start from cash. No stitched or independent-seed market claim.','folds':rows}
(R/'reports/static_chronology_summary.json').write_text(json.dumps(result,indent=2))
lines=['# Static native LoRA: chronological robustness','',result['scope'],'','| Evaluation | Base Sharpe | Trained Sharpe | Gain | Positive seeds | Paired 95% interval |','|---|---:|---:|---:|---:|---|']
for r in rows:lines.append(f"| {r['period']} | {r['sharpe_proxy']['initial']:.4f} | {r['sharpe_proxy']['trained']:.4f} | {r['mean_Sharpe_gain']:+.4f} | {r['positive_seeds']}/3 | [{r['interval20'][0]:.4f}, {r['interval20'][1]:.4f}] |")
lines+=['','Static preference learning only. Small cross-seed variation is expected for this low-dimensional decision problem; it does not replace uncertainty across market periods. No reserved2026 portfolio evaluated.']
(R/'reports/STATIC_CHRONOLOGY.md').write_text('\n'.join(lines));print('\n'.join(lines))
