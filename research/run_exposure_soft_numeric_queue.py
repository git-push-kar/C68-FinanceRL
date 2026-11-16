"""Serial controlled actor-fitting comparison, CPU only."""
import json,os,subprocess,sys,time
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).parent;status=ROOT/'reports/exposure_soft_numeric_queue_status.json';results=[]
env=dict(os.environ);env.update(CUDA_VISIBLE_DEVICES='-1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
def read(path):
    try:return json.loads(path.read_text())
    except (FileNotFoundError,json.JSONDecodeError):return {}
def write(**kw):status.write_text(json.dumps({'pid':os.getpid(),'device':'cpu','results':results,**kw},indent=2))
def command(args):
    r=subprocess.run([sys.executable,'-u',*args],cwd=ROOT,env=env,creationflags=subprocess.CREATE_NO_WINDOW)
    if r.returncode:
        write(state='needs_review',reason=f'Exit {r.returncode}: {args}');raise SystemExit(r.returncode)
def launch(name,args):command(['record_launch.py','--name',name,'--source','rl_exposure_soft_trial','--',*args])
write(state='waiting_for_exposure_numeric')
while True:
    upstream=read(ROOT/'reports/exposure_numeric_queue_status.json')
    if upstream.get('state')=='completed_development_only':break
    if upstream.get('state')=='needs_review':
        write(state='needs_review',reason='Upstream CPU experiment failed');raise SystemExit(1)
    time.sleep(20)
write(state='smoke_testing')
launch('sector_exposure_soft_numeric_smoke',['train_numeric.py','--output','../checks/exposure_soft_numeric_smoke',
       '--policy-lr','.001','--updates','2','--eval-every','2'])
launch('audit_sector_exposure_soft_numeric_smoke',['audit_numeric.py','--run','../checks/exposure_soft_numeric_smoke'])
smoke=ROOT/'checks/exposure_soft_numeric_smoke'
if read(smoke/'reload_audit.json')['status']!='passed':raise AssertionError('Smoke failed')
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
(ROOT/'checks/exposure_actor_loss_initialization_parity.json').write_text(json.dumps({'status':'passed','identical_weights':True,'checks':checks},indent=2))
for seed in (42,123,456):
    name=f'sector_exposure_soft_numeric_seed{seed}_1024';run=ROOT/'campaign_runs'/name
    write(state='running',job=name)
    launch(name,['train_numeric.py','--output','../campaign_runs/'+name,'--policy-lr','.001','--seed',str(seed),'--updates','1024','--eval-every','32'])
    launch('audit_'+name,['audit_numeric.py','--run','../campaign_runs/'+name])
    launch('adaptation_'+name,['controls.py','--run','../campaign_runs/'+name])
    command(['compare_rl_learning.py','--run',str(run)]);results.append(name)
    command(['collect_rl_evidence.py']);command(['backup_rl_progress.py'])
command(['compare_rl_seeds.py','--runs',*[str(ROOT/'campaign_runs'/n) for n in results],
         '--budget','1024','--output',str(ROOT/'reports/exposure_soft_numeric_seed_comparison_budget1024.json')])
command(['backup_rl_progress.py']);write(state='completed_development_only',note='CPU policy-iteration comparison, not an InternVL adapter; no reserved test.')
