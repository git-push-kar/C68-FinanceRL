import json,os,sys,subprocess
from pathlib import Path
R=Path(__file__).parent;results=[];status=R/'reports/multiasset_controls_status.json'
env=dict(os.environ);env.update(CUDA_VISIBLE_DEVICES='-1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
def write(**kw):status.write_text(json.dumps({'pid':os.getpid(),'results':results,**kw},indent=2))
def cmd(args):subprocess.run([sys.executable,'-u',*args],cwd=R,env=env,check=True,creationflags=subprocess.CREATE_NO_WINDOW)
def launch(source,name,args):
 write(state='running',job=name);cmd(['record_launch.py','--name',name,'--source',source,'--',*args])
try:
 for variant in ('static','dynamic'):
  source='rl_multiasset_'+variant;names=[]
  for seed in (42,123,456):
   name=f'multiasset_{variant}_seed{seed}_64';names.append(name)
   launch(source,name,['train_direct.py','--output','../campaign_runs/'+name,'--seed',str(seed)])
   launch(source,'audit_'+name,['audit_numeric.py','--run','../campaign_runs/'+name])
   launch(source,'controls_'+name,['controls.py','--run','../campaign_runs/'+name]);results.append(name)
  cmd(['compare_rl_seeds.py','--runs',*[str(R/'campaign_runs'/n) for n in names],'--budget','64','--output',str(R/f'reports/multiasset_{variant}_comparison64.json')])
 write(state='completed_development_only')
except Exception as exc:write(state='needs_review',reason=repr(exc));raise
