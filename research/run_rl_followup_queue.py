"""Persist a single-GPU queue across interactive status questions."""
import json,os,subprocess,sys,time
from datetime import datetime,timezone
from pathlib import Path

ROOT=Path(__file__).parent
status_path=ROOT/'reports/rl_followup_queue_status.json'
def status(**fields):
    status_path.write_text(json.dumps({'observed_utc':datetime.now(timezone.utc).isoformat(),**fields},indent=2))

parent=ROOT/'launch_records/strategy_ppo_internvl_seed42_256_persistent/status.json'
status(state='waiting',dependency=str(parent),pid=os.getpid())
while True:
    if parent.exists():
        try:d=json.loads(parent.read_text())
        except json.JSONDecodeError:time.sleep(5);continue
        if d['status']=='completed':break
        if d['status']=='failed':
            status(state='blocked_by_failed_dependency',dependency=str(parent),pid=os.getpid());sys.exit(1)
    time.sleep(15)
env=dict(os.environ);env.update(FINANCERL_EXECUTION='selective_fp32',
    FINANCERL_BASE_MODEL=str(ROOT/'local_models/InternVL3-2B'),HF_HUB_OFFLINE='1',
    TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
jobs=[('strategy_ppo_reload_extended','rl_strategy_trial',['audit_strategy.py','--run','../campaign_runs/strategy_ppo_internvl_seed42_256_persistent']),
      ('sector_trajectory_grpo_internvl_seed42_64','rl_sector_trial',
       ['trajectory_grpo.py','--output','../campaign_runs/sector_trajectory_grpo_internvl_seed42_64','--updates','64','--eval-every','16'])]
for name,source,args in jobs:
    status(state='running',job=name,pid=os.getpid())
    result=subprocess.run([sys.executable,'-u',str(ROOT/'record_launch.py'),'--name',name,'--source',source,'--',*args],
        cwd=ROOT,env=env,creationflags=subprocess.CREATE_NO_WINDOW)
    if result.returncode:
        status(state='failed',job=name,exit_code=result.returncode,pid=os.getpid());sys.exit(result.returncode)
status(state='completed',pid=os.getpid())
