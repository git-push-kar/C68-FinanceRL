"""One native five-strategy experiment, serialized after existing GPU ablations."""
import json,os,subprocess,sys,time
from pathlib import Path
R=Path(__file__).parent;results=[];status=R/'reports/risk_selector_native_status.json'
env=dict(os.environ);env.update(FINANCERL_EXECUTION='selective_fp32',FINANCERL_BASE_MODEL=str(R/'local_models/InternVL3-2B'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2');env.pop('CUDA_VISIBLE_DEVICES',None)

def read(p):return json.loads(p.read_text())
def write(**kw):status.write_text(json.dumps({'pid':os.getpid(),'results':results,**kw},indent=2))
def cmd(args,cpu=False):
 child_env=dict(env)
 if cpu:child_env['CUDA_VISIBLE_DEVICES']='-1'
 subprocess.run([sys.executable,'-u',*args],cwd=R,env=child_env,check=True,creationflags=subprocess.CREATE_NO_WINDOW)
def launch(name,args):
 write(state='running',job=name);cmd(['record_launch.py','--name',name,'--source','rl_risk_selector_native','--',*args])
try:
 write(state='waiting_for_readout_GPU_and_risk_selector_CPU')
 while True:
  gpu=read(R/'reports/readout_only_ablation_status.json');cpu=read(R/'reports/risk_selector_controls_status.json')
  if any(x['state']=='needs_review' for x in (gpu,cpu)):raise RuntimeError('Upstream requires review')
  if all(x['state']=='completed_development_only' for x in (gpu,cpu)):break
  time.sleep(20)
 if not read(R/'reports/risk_selector_screen.json')['arms']['static']['eligible_for_native_review']:
  write(state='completed_no_promotion',reason='Declared CPU static screen failed');sys.exit(0)
 launch('risk_selector_native_gate',['text_gate.py'])
 gate=read(R/'checks/risk_selector_native_gate/result.json')
 if gate['status']!='passed' or gate['frozen_before']!=read(R/'checks/multiasset_native_gate/result.json')['frozen_before']:
  raise AssertionError('Technical gate or pristine base identity differs')
 for seed in (42,123,456):
  name=f'risk_selector_native_seed{seed}_64';run=R/'campaign_runs'/name
  launch(name,['train_static.py','--output','../campaign_runs/'+name,'--seed',str(seed)])
  launch('audit_'+name,['audit_static.py','--run','../campaign_runs/'+name])
  launch('endpoint_'+name,['audit_static.py','--run','../campaign_runs/'+name,'--endpoint'])
  launch('controls_'+name,['controls_static.py','--run','../campaign_runs/'+name])
  cmd(['audit_static_sampling_cpu.py','--run',str(run),'--source',str(R/'rl_risk_selector_native'),
       '--data',str(R/'multiasset_data_20261115/five_asset_development_through2024.csv'),
       '--start','2022-01-01','--end','2024-12-31'],cpu=True)
  results.append(name)
 cmd(['compare_rl_seeds.py','--runs',*[str(R/'campaign_runs'/n) for n in results],
      '--budget','64','--output',str(R/'reports/risk_selector_native_comparison64.json')],cpu=True)
 write(state='completed_development_only')
except Exception as exc:write(state='needs_review',reason=repr(exc));raise
