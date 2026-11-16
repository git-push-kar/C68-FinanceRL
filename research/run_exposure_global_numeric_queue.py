"""CPU full-history direct-reinforcement feasibility, serialized after actor-fitting controls."""
import json,os,subprocess,sys,time
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).parent;status=ROOT/'reports/exposure_global_numeric_queue_status.json';results=[]
env=dict(os.environ);env.update(CUDA_VISIBLE_DEVICES='-1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
def read(path):
    try:return json.loads(path.read_text())
    except (FileNotFoundError,json.JSONDecodeError):return {}
def write(**kw):status.write_text(json.dumps({'pid':os.getpid(),'device':'cpu','results':results,**kw},indent=2))
def command(args):
    r=subprocess.run([sys.executable,'-u',*args],cwd=ROOT,env=env,creationflags=subprocess.CREATE_NO_WINDOW)
    if r.returncode:raise RuntimeError(f'Exit {r.returncode}: {args}')
def launch(name,args):command(['record_launch.py','--name',name,'--source','rl_exposure_global_trial','--',*args])
def main():
    smoke=ROOT/'checks/exposure_global_numeric_smoke'
    if read(smoke/'reload_audit.json').get('status')!='passed':raise AssertionError('Smoke audit required')
    reference=ROOT/'campaign_runs/sector_exposure_numeric_seed42_1024'
    a=torch.load(reference/'checkpoint_0/numeric.pt',map_location='cpu',weights_only=True)
    b=torch.load(smoke/'checkpoint_0/numeric.pt',map_location='cpu',weights_only=True)
    if a.keys()!=b.keys() or any(not torch.equal(a[k],b[k]) for k in a):raise AssertionError('Different initialization')
    checks=[]
    for split in ('validation','development'):
        x=read(reference/f'{split}_0.json');y=read(smoke/f'{split}_0.json')
        error=float(np.max(abs(np.asarray(x['probabilities'])-np.asarray(y['probabilities']))))
        if error!=0 or x['daily']!=y['daily']:raise AssertionError('Different initial trajectory')
        checks.append({'split':split,'maximum_probability_error':error,'identical_daily':True})
    (ROOT/'checks/exposure_global_initialization_parity.json').write_text(json.dumps({'status':'passed','identical_weights':True,'checks':checks},indent=2))
    write(state='waiting_for_exposure_direct_numeric')
    while True:
        upstream=read(ROOT/'reports/exposure_direct_numeric_queue_status.json')
        if upstream.get('state')=='completed_development_only':break
        if upstream.get('state')=='needs_review':raise RuntimeError('Upstream experiment failed')
        time.sleep(20)
    for seed in (42,123,456):
        name=f'sector_exposure_global_numeric_seed{seed}_64';run=ROOT/'campaign_runs'/name
        write(state='running',job=name)
        launch(name,['train_direct.py','--output','../campaign_runs/'+name,'--seed',str(seed),'--updates','64','--eval-every','4'])
        launch('audit_'+name,['audit_numeric.py','--run','../campaign_runs/'+name])
        launch('adaptation_'+name,['controls.py','--run','../campaign_runs/'+name])
        command(['compare_rl_learning.py','--run',str(run)]);results.append(name)
        command(['collect_rl_evidence.py']);command(['backup_rl_progress.py'])
    command(['compare_rl_seeds.py','--runs',*[str(ROOT/'campaign_runs'/n) for n in results],
             '--budget','64','--output',str(ROOT/'reports/exposure_global_numeric_seed_comparison_budget64.json')])
    command(['backup_rl_progress.py']);write(state='completed_development_only',note='CPU full-history direct RL feasibility only; not an InternVL adapter; no reserved test.')
try:main()
except Exception as exc:
    write(state='needs_review',reason=repr(exc));raise
