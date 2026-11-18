"""Declared CPU feasibility experiment; never starts a native/GPU job."""
import json,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).parent;status=ROOT/'reports/exposure_numeric_queue_status.json';results=[]
env=dict(os.environ);env.update(CUDA_VISIBLE_DEVICES='-1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
def write(**kw):status.write_text(json.dumps({'pid':os.getpid(),'device':'cpu','results':results,**kw},indent=2))
def command(args):
    r=subprocess.run([sys.executable,'-u',*args],cwd=ROOT,env=env,creationflags=subprocess.CREATE_NO_WINDOW)
    if r.returncode:
        write(state='needs_review',reason=f'Exit {r.returncode}: {args}');raise SystemExit(r.returncode)
smoke=json.loads((ROOT/'checks/exposure_numeric_smoke/reload_audit.json').read_text())
if smoke['status']!='passed' or smoke['reload_max_probability_error']>1e-7:raise AssertionError('Smoke audit required')
write(state='starting')
command(['record_launch.py','--name','exposure_declared_controls','--source','rl_exposure_trial','--','controls.py','--output','../reports/exposure_declared_controls.json'])
for seed in (42,123,456):
    name=f'sector_exposure_numeric_seed{seed}_1024';run=ROOT/'campaign_runs'/name
    write(state='running',job=name)
    command(['record_launch.py','--name',name,'--source','rl_exposure_trial','--','train_numeric.py',
             '--output','../campaign_runs/'+name,'--policy-lr','.001','--seed',str(seed),'--updates','1024','--eval-every','32'])
    command(['record_launch.py','--name','audit_'+name,'--source','rl_exposure_trial','--','audit_numeric.py','--run','../campaign_runs/'+name])
    command(['record_launch.py','--name','adaptation_'+name,'--source','rl_exposure_trial','--','controls.py','--run','../campaign_runs/'+name])
    command(['compare_rl_learning.py','--run',str(run)])
    results.append(name);command(['backup_rl_progress.py'])
command(['compare_rl_seeds.py','--runs',*[str(ROOT/'campaign_runs'/n) for n in results],
         '--budget','1024','--output',str(ROOT/'reports/exposure_numeric_seed_comparison_budget1024.json')])
command(['backup_rl_progress.py']);write(state='completed_development_only',note='CPU feasibility evidence only; not an InternVL adapter. No reserved test.')
