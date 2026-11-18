import os,json,subprocess,sys,time
from pathlib import Path
R=Path(__file__).parent;status=R/'reports/multiasset_native_status.json';results=[]
env=dict(os.environ);env.update(FINANCERL_EXECUTION='selective_fp32',FINANCERL_BASE_MODEL=str(R/'local_models/InternVL3-2B'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
env.pop('CUDA_VISIBLE_DEVICES',None)
def write(**kw):status.write_text(json.dumps({'pid':os.getpid(),'results':results,**kw},indent=2))
def command(args):
 p=subprocess.run([sys.executable,'-u',*args],cwd=R,env=env,creationflags=subprocess.CREATE_NO_WINDOW)
 if p.returncode:raise RuntimeError(str(args))
def launch(name,args):
 write(state='running',job=name);command(['record_launch.py','--name',name,'--source','rl_multiasset_native','--',*args])
try:
 write(state='waiting_for_static_CPU_controls')
 while not (R/'reports/multiasset_static_comparison64.json').exists():
  upstream=json.loads((R/'reports/multiasset_controls_status.json').read_text())
  if upstream['state']=='needs_review':raise RuntimeError('CPU controls failed')
  time.sleep(20)
 screen=json.loads((R/'reports/multiasset_static_comparison64.json').read_text())
 if screen['mean_Sharpe_gain']<=0 or screen['positive_seeds']<2:
  write(state='completed_no_promotion',reason='CPU feasibility screen failed');sys.exit(0)
 launch('multiasset_native_gate',['text_gate.py'])
 if json.loads((R/'checks/multiasset_native_gate/result.json').read_text())['status']!='passed':raise AssertionError('Gate failed')
 for seed in (42,123,456):
  name=f'multiasset_native_seed{seed}_64'
  launch(name,['train_static.py','--output','../campaign_runs/'+name,'--seed',str(seed)])
  launch('audit_'+name,['audit_static.py','--run','../campaign_runs/'+name]);results.append(name)
  command(['compare_rl_learning.py','--run',str(R/'campaign_runs'/name)])
 command(['compare_rl_seeds.py','--runs',*[str(R/'campaign_runs'/n) for n in results],'--budget','64','--output',str(R/'reports/multiasset_native_comparison64.json')])
 write(state='completed_development_only')
except Exception as exc:write(state='needs_review',reason=repr(exc));raise
