import json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).parent;status=ROOT/'reports/candidate_static_queue_status.json';results=[]
env=dict(os.environ);env.update(CUDA_VISIBLE_DEVICES='-1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
def read(p):return json.loads(p.read_text())
def write(**kw):status.write_text(json.dumps({'pid':os.getpid(),'results':results,**kw},indent=2))
def command(args):
 r=subprocess.run([sys.executable,'-u',*args],cwd=ROOT,env=env,creationflags=subprocess.CREATE_NO_WINDOW)
 if r.returncode:raise RuntimeError(f'Exit {r.returncode}: {args}')
def launch(name,args):command(['record_launch.py','--name',name,'--source','rl_candidate_static_trial','--',*args])
def main():
 write(state='waiting_for_walkforward')
 while True:
  state=read(ROOT/'reports/exposure_walkforward_queue_status.json')['state']
  if state=='completed_development_only':break
  if state=='needs_review':raise RuntimeError('Upstream CPU run failed')
  time.sleep(20)
 launch('candidate_static_smoke',['train_direct.py','--output','../checks/candidate_static_smoke','--updates','2','--eval-every','2'])
 launch('audit_candidate_static_smoke',['audit_numeric.py','--run','../checks/candidate_static_smoke'])
 if read(ROOT/'checks/candidate_static_smoke/reload_audit.json')['status']!='passed':raise AssertionError('Reload gate failed')
 for seed in (42,123,456):
  name=f'nifty_candidate_static_seed{seed}_64';run=ROOT/'campaign_runs'/name;write(state='running',job=name)
  launch(name,['train_direct.py','--output','../campaign_runs/'+name,'--seed',str(seed),'--updates','64','--eval-every','4'])
  launch('audit_'+name,['audit_numeric.py','--run','../campaign_runs/'+name])
  launch('controls_'+name,['controls.py','--run','../campaign_runs/'+name])
  command(['compare_rl_learning.py','--run',str(run)]);results.append(name)
 command(['compare_rl_seeds.py','--runs',*[str(ROOT/'campaign_runs'/n) for n in results],'--budget','64',
  '--output',str(ROOT/'reports/candidate_static_seed_comparison_budget64.json')])
 command(['backup_rl_progress.py']);write(state='completed_development_only',note='CPU feasibility control, not native adapter; all seeds retained')
try:main()
except Exception as exc:write(state='needs_review',reason=repr(exc));raise
