"""Run the next documented RL hypothesis after existing GPU audits finish."""
import json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).parent;dependency=ROOT/'reports/rl_audit_queue_status.json'
status=ROOT/'reports/counterfactual_queue_status.json'
def write(**k):status.write_text(json.dumps({'pid':os.getpid(),**k},indent=2))
write(state='waiting',dependency=str(dependency))
while True:
    try:d=json.loads(dependency.read_text())
    except (FileNotFoundError,json.JSONDecodeError):time.sleep(15);continue
    if d['state'] in ('completed','completed_with_failures'):break
    time.sleep(15)
env=dict(os.environ);env.update(FINANCERL_EXECUTION='selective_fp32',FINANCERL_BASE_MODEL=str(ROOT/'local_models/InternVL3-2B'),
    HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
run='sector_native_counterfactual_seed42_64'
jobs=[(run,['train_counterfactual.py','--output','../campaign_runs/'+run,'--updates','64','--eval-every','16']),
      ('audit_'+run,['audit_text_run.py','--run','../campaign_runs/'+run])]
for name,args in jobs:
    write(state='running',job=name)
    r=subprocess.run([sys.executable,'-u',str(ROOT/'record_launch.py'),'--name',name,'--source','rl_counterfactual_trial','--',*args],
        cwd=ROOT,env=env,creationflags=subprocess.CREATE_NO_WINDOW)
    if r.returncode:write(state='failed',job=name,exit_code=r.returncode);sys.exit(r.returncode)
write(state='completed')
