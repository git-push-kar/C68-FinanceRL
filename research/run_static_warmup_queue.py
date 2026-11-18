import os,json,subprocess,sys,time
from pathlib import Path
R=Path(__file__).parent;status=R/'reports/static_warmup_queue_status.json';results=[]
env=dict(os.environ);env.update(FINANCERL_EXECUTION='selective_fp32',FINANCERL_BASE_MODEL=str(R/'local_models/InternVL3-2B'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
env.pop('CUDA_VISIBLE_DEVICES',None)
def write(**kw):status.write_text(json.dumps({'pid':os.getpid(),'results':results,**kw},indent=2))
def command(args):
 p=subprocess.run([sys.executable,'-u',*args],cwd=R,env=env,creationflags=subprocess.CREATE_NO_WINDOW)
 if p.returncode:raise RuntimeError(str(args))
def launch(name,args):
 write(state='running',job=name);command(['record_launch.py','--name',name,'--source','rl_static_warmup','--',*args])
try:
 deadline=time.monotonic()+600
 write(state='waiting_for_training_only_target')
 while not (R/'reports/training_allocation_control/results.json').exists():
  if time.monotonic()>deadline:raise TimeoutError('Training-only target unavailable')
  time.sleep(10)
 for seed in (42,123,456):
  name=f'nifty_warmup_native_seed{seed}_64'
  launch(name,['train_static.py','--output','../campaign_runs/'+name,'--seed',str(seed)])
  launch('audit_'+name,['audit_static.py','--run','../campaign_runs/'+name]);results.append(name)
  command(['compare_rl_learning.py','--run',str(R/'campaign_runs'/name)])
 command(['compare_rl_seeds.py','--runs',*[str(R/'campaign_runs'/n) for n in results],'--budget','64','--output',str(R/'reports/warmup_native_seed_comparison_budget64.json')])
 write(state='completed_development_only')
except Exception as exc:write(state='needs_review',reason=repr(exc));raise
