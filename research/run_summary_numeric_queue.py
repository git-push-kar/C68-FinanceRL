"""CPU-only prompt-information control; validation-selected bounded calibration."""
import json,os,subprocess,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).parent;status=ROOT/'reports/summary_numeric_queue_status.json';results=[]
env=dict(os.environ);env.update(CUDA_VISIBLE_DEVICES='-1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
def write(**values):status.write_text(json.dumps({'pid':os.getpid(),'device':'cpu','results':results,**values},indent=2))
def read(p):return json.loads(p.read_text())
def command(args):
    r=subprocess.run([sys.executable,'-u',*args],cwd=ROOT,env=env,creationflags=subprocess.CREATE_NO_WINDOW)
    if r.returncode:
        write(state='needs_review',reason=f'Exit {r.returncode}: {args}');raise SystemExit(r.returncode)
def train_audit(lr,seed,budget):
    tag=str(lr).replace('.','p');name=f'sector_summary_numeric_lr{tag}_seed{seed}_{budget}'
    write(state='running',job=name)
    command(['record_launch.py','--name',name,'--source','rl_summary_numeric_control','--','train_numeric.py',
             '--output','../campaign_runs/'+name,'--policy-lr',str(lr),'--seed',str(seed),'--updates',str(budget),'--eval-every','16'])
    command(['record_launch.py','--name','audit_'+name,'--source','rl_summary_numeric_control','--','audit_numeric.py',
             '--run','../campaign_runs/'+name])
    run=ROOT/'campaign_runs'/name
    if read(run/'reload_audit.json')['status']!='passed':raise AssertionError('Audit failed')
    command(['compare_rl_learning.py','--run',str(run)])
    results.append({'run':name,'lr':lr,'seed':seed,'budget':budget})
    return name,read(run/'selection.json')['selected']
if read(ROOT/'checks/summary_numeric_smoke/reload_audit.json')['status']!='passed':raise AssertionError('Smoke audit required')
write(state='starting');grid=[]
for lr in (1e-4,3e-4,1e-3):
    rows=[]
    for seed in (42,123,456):
        name,selected=train_audit(lr,seed,256)
        rows.append({'run':name,'seed':seed,'selected_step':selected['step'],'validation':selected['validation'],
                     'development':selected['development']})
    grid.append({'lr':lr,'mean_validation_Sharpe':float(np.mean([r['validation']['sharpe_proxy'] for r in rows])),'runs':rows})
chosen=max(grid,key=lambda g:g['mean_validation_Sharpe'])
report={'scope':'Development-only bounded calibration; distinct initial policy from InternVL',
    'selection':'One rate by mean validation Sharpe across three seeds; lower rate wins exact ties',
    'selected_lr':chosen['lr'],'grid':grid,'reserved_test_accessed':False}
(ROOT/'reports/summary_numeric_calibration.json').write_text(json.dumps(report,indent=2))
command(['compare_rl_seeds.py','--runs',*[str(ROOT/'campaign_runs'/r['run']) for r in chosen['runs']],
         '--budget','256','--output',str(ROOT/'reports/summary_numeric_seed_comparison_budget256.json')])
command(['collect_rl_evidence.py']);command(['backup_rl_progress.py'])
# Stronger cheap baseline in addition to the sample-budget-matched comparison.
long=[]
for seed in (42,123,456):
    name,_=train_audit(chosen['lr'],seed,1024);long.append(name)
command(['compare_rl_seeds.py','--runs',*[str(ROOT/'campaign_runs'/n) for n in long],
         '--budget','1024','--output',str(ROOT/'reports/summary_numeric_seed_comparison_budget1024.json')])
command(['collect_rl_evidence.py']);command(['backup_rl_progress.py'])
write(state='completed_development_only',selected_lr=chosen['lr'],extended_runs=long,
      note='1024-update control is not sample-budget matched to InternVL; report separately.')
