"""Write a compact local status page from recorded evidence, not predictions."""
from datetime import datetime
import json
from pathlib import Path
import subprocess
from execution_policy import IST,quiet_now

ROOT=Path(__file__).parent
rows=[]
for folder in sorted((ROOT/'campaign_runs').glob('*')):
    if not folder.is_dir():continue
    if (folder/'selection.json').exists():
        result=json.loads((folder/'selection.json').read_text());selected=result.get('selected',{})
        rows.append({'run':folder.name,'state':'completed selection','selected_step':selected.get('step'),
                     'metrics':selected.get('metrics',{})})
    elif (folder/'learning_check.json').exists():
        result=json.loads((folder/'learning_check.json').read_text())
        rows.append({'run':folder.name,'state':'completed training','updates':result.get('optimizer_updates')})
    elif (folder/'summary.json').exists():rows.append({'run':folder.name,'state':'completed comparison'})
    elif (folder/'result.json').exists():rows.append({'run':folder.name,'state':'completed control'})
    else:
        losses=json.loads((folder/'losses.json').read_text()) if (folder/'losses.json').exists() else []
        rows.append({'run':folder.name,'state':'in progress / inspect process and log','last_record':losses[-1] if losses else None})
gpu=subprocess.run(['nvidia-smi','--query-gpu=memory.used,utilization.gpu,temperature.gpu,fan.speed,power.draw',
    '--format=csv,noheader'],capture_output=True,text=True).stdout.strip()
final_path=ROOT/'campaign_runs/reserved_post_publication_c4f2f205fb74/status.json'
final_state=json.loads(final_path.read_text()) if final_path.exists() else {'status':'not started'}
result={'observed_ist':datetime.now(IST).isoformat(),'quiet_window':quiet_now(),'gpu':gpu,'runs':rows,
    'frozen_final_evaluation':final_state,
    'constraints':'No heavy compute 22:50–07:00 IST; fixed pristine InternVL3-2B; final May-2025–Jan-2026 block uses frozen models and one-close signal delay',
    'next':('Final evaluation complete. Read reports/FINAL_REPORT.md and the portable archive verification receipt; no post-test model selection or training.' if final_state['status']=='completed' else 'Complete the already frozen final evaluation, then analyze and verify the portable bundle. No method or checkpoint changes.')}
(ROOT/'STATUS.json').write_text(json.dumps(result,indent=2))
lines=['# Finance campaign status','',f"Observed: {result['observed_ist']}",'',result['constraints'],'',
       'GPU (memory, utilization, temperature, fan, power): '+gpu,'',
       '| Run | Recorded state |','|---|---|']
lines.extend(f"| {r['run']} | {r['state']} |" for r in rows)
lines.extend(['','This is a snapshot, not a claim of continuous unattended monitoring.',
    'See CAMPAIGN.md, LITERATURE_COMPARISON.md and individual run records for decisions and evidence.'])
(ROOT/'STATUS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps(result,indent=2))
