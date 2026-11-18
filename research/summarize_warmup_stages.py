import json
from pathlib import Path
import numpy as np
R=Path(__file__).parent;rows=[]
for seed in (42,123,456):
 r=R/f'campaign_runs/nifty_warmup_native_seed{seed}_64';s=json.loads((r/'selection.json').read_text());raw=json.loads((r/'raw_development.json').read_text());old=json.loads((R/f'campaign_runs/nifty_static_native_seed{seed}_64/selection.json').read_text())
 rows.append({'seed':seed,'selected_step':s['selected']['step'],'raw':raw['metrics'],'warmup':s['initial']['development'],'warmup_plus_RL':s['selected']['development'],'previous_direct_RL':old['selected']['development']})
means={name:float(np.mean([r[name]['sharpe_proxy'] for r in rows])) for name in ['raw','warmup','warmup_plus_RL','previous_direct_RL']}
result={'scope':'Reused development, validation-only checkpoints; warm-up target chosen only by training objective. No reserved outcome.','mean_Sharpe':means,'incremental_RL_gain':means['warmup_plus_RL']-means['warmup'],'gain_over_prior_direct_RL':means['warmup_plus_RL']-means['previous_direct_RL'],'runs':rows}
(R/'reports/warmup_stage_comparison.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='runs'},indent=2))
