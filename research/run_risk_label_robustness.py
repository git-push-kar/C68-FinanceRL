import json,os,sys,subprocess,time
from pathlib import Path
R=Path(__file__).parent;status=R/'reports/risk_label_robustness_status.json';results=[]
env=dict(os.environ);env.update(FINANCERL_EXECUTION='selective_fp32',FINANCERL_BASE_MODEL=str(R/'local_models/InternVL3-2B'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
env.pop('CUDA_VISIBLE_DEVICES',None)

def write(**kw):status.write_text(json.dumps({'pid':os.getpid(),'results':results,**kw},indent=2))
def cmd(args):subprocess.run([sys.executable,'-u',*args],cwd=R,env=env,check=True,creationflags=subprocess.CREATE_NO_WINDOW)
def launch(source,name,args):
 write(state='running',job=name);cmd(['record_launch.py','--name',name,'--source',source,'--',*args])
try:
 write(state='waiting_for_risk_selector_native_GPU')
 while True:
  upstream=json.loads((R/'reports/risk_selector_native_status.json').read_text())
  if upstream['state']=='completed_development_only':break
  if upstream['state']=='completed_no_promotion':
   write(state='completed_no_followup',reason='Original risk selector was not promoted');sys.exit(0)
  if upstream['state']=='needs_review':raise RuntimeError('Upstream GPU needs review')
  time.sleep(20)
 screen=json.loads((R/'reports/risk_selector_native_comparison64.json').read_text())
 if screen['mean_Sharpe_gain']<=0 or screen['positive_seeds']<2:
  write(state='completed_no_followup',reason='Original native risk selector did not improve repeatably');sys.exit(0)
 for variant in ('rotate','reverse'):
  source='rl_risk_labels_'+variant;names=[]
  for seed in (42,123,456):
   name=f'risk_labels_{variant}_seed{seed}_64';names.append(name)
   launch(source,name,['train_static.py','--output','../campaign_runs/'+name,'--seed',str(seed)])
   launch(source,'audit_'+name,['audit_static.py','--run','../campaign_runs/'+name])
   launch(source,'endpoint_'+name,['audit_static.py','--run','../campaign_runs/'+name,'--endpoint']);results.append(name)
  cmd(['compare_rl_seeds.py','--runs',*[str(R/'campaign_runs'/n) for n in names],'--budget','64','--output',str(R/f'reports/risk_label_{variant}_comparison64.json')])
 write(state='completed_development_only')
except Exception as exc:write(state='needs_review',reason=repr(exc));raise
