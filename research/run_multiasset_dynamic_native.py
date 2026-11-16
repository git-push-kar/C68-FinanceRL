import json,os,sys,subprocess,time
from pathlib import Path
R=Path(__file__).parent;status=R/'reports/multiasset_dynamic_native_status.json';results=[]
env=dict(os.environ);env.update(FINANCERL_EXECUTION='selective_fp32',FINANCERL_BASE_MODEL=str(R/'local_models/InternVL3-2B'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2');env.pop('CUDA_VISIBLE_DEVICES',None)
def read(p):return json.loads(p.read_text())
def write(**kw):status.write_text(json.dumps({'pid':os.getpid(),'results':results,**kw},indent=2))
def cmd(args):subprocess.run([sys.executable,'-u',*args],cwd=R,env=env,check=True,creationflags=subprocess.CREATE_NO_WINDOW)
def launch(name,args):
 write(state='running',job=name);cmd(['record_launch.py','--name',name,'--source','rl_multiasset_dynamic_native','--',*args])
try:
 write(state='waiting_for_static_GPU_and_dynamic_CPU')
 while True:
  gpu=read(R/'reports/multiasset_native_status.json');cpu=read(R/'reports/multiasset_controls_status.json')
  if any(s['state']=='needs_review' for s in (gpu,cpu)):raise RuntimeError('Upstream needs review')
  if gpu['state']=='completed_development_only' and cpu['state']=='completed_development_only':break
  time.sleep(20)
 screen=read(R/'reports/multiasset_dynamic_comparison64.json')
 if screen['mean_Sharpe_gain']<=0 or screen['positive_seeds']<2:
  write(state='completed_no_promotion',reason='Dynamic CPU screen failed');sys.exit(0)
 launch('multiasset_dynamic_native_gate',['text_gate.py'])
 if read(R/'checks/multiasset_dynamic_native_gate/result.json')['status']!='passed':raise AssertionError('Gate failed')
 launch('multiasset_dynamic_native_smoke',['train_direct.py','--output','../checks/multiasset_dynamic_native_smoke','--updates','2','--eval-every','2'])
 launch('audit_multiasset_dynamic_native_smoke',['audit_text_run.py','--run','../checks/multiasset_dynamic_native_smoke'])
 for seed in (42,123,456):
  name=f'multiasset_dynamic_native_seed{seed}_16'
  launch(name,['train_direct.py','--output','../campaign_runs/'+name,'--seed',str(seed),'--updates','16','--eval-every','4'])
  launch('audit_'+name,['audit_text_run.py','--run','../campaign_runs/'+name]);results.append(name)
 cmd(['compare_rl_seeds.py','--runs',*[str(R/'campaign_runs'/n) for n in results],'--budget','16','--output',str(R/'reports/multiasset_dynamic_native_comparison16.json')])
 write(state='completed_development_only')
except Exception as exc:write(state='needs_review',reason=repr(exc));raise
