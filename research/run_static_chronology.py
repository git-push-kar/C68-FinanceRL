import json,os,sys,subprocess
from pathlib import Path
import numpy as np
R=Path(__file__).parent;results=[];status=R/'reports/static_chronology_status.json'
env=dict(os.environ);env.update(FINANCERL_EXECUTION='selective_fp32',FINANCERL_BASE_MODEL=str(R/'local_models/InternVL3-2B'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
def read(p):return json.loads(p.read_text())
def write(**kw):status.write_text(json.dumps({'pid':os.getpid(),'results':results,**kw},indent=2))
def cmd(args):subprocess.run([sys.executable,'-u',*args],cwd=R,env=env,check=True,creationflags=subprocess.CREATE_NO_WINDOW)
def launch(name,args):
 write(state='running',job=name);cmd(['record_launch.py','--name',name,'--source','rl_static_chronology','--',*args])
try:
 launch('static_chronology_smoke',['train_static.py','--output','../checks/static_chronology_smoke','--updates','4','--eval-every','4'])
 for step in (0,4):
  for split in ('validation','development'):
   a=read(R/f'campaign_runs/nifty_static_native_seed42_64/{split}_{step}.json');b=read(R/f'checks/static_chronology_smoke/{split}_{step}.json')
   if a!=b:raise AssertionError('Date-only parameterization changed original trajectory')
 (R/'checks/static_chronology_prefix_parity.json').write_text(json.dumps({'status':'passed','steps':[0,4],'identical_evaluations':True}))
 for fold,end,val,dev in [(1,'2019-12-31',2020,2021),(2,'2020-12-31',2021,2022)]:
  names=[]
  for seed in (42,123,456):
   name=f'nifty_static_f{fold}_seed{seed}_64';names.append(name)
   launch(name,['train_static.py','--output','../campaign_runs/'+name,'--seed',str(seed),'--train-end',end,'--validation-year',str(val),'--development-year',str(dev),'--development-end',f'{dev}-12-31'])
   launch('audit_'+name,['audit_static.py','--run','../campaign_runs/'+name]);results.append(name)
  cmd(['compare_rl_seeds.py','--runs',*[str(R/'campaign_runs'/n) for n in names],'--budget','64','--output',str(R/f'reports/static_chronology_f{fold}.json')])
 write(state='completed_development_only')
except Exception as exc:write(state='needs_review',reason=repr(exc));raise
