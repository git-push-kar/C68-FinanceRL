"""Independent seed replications after the serial pilot queue releases GPU."""
import json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).parent;dependency=ROOT/'reports/native_text_queue_status.json'
status=ROOT/'reports/rl_replication_queue_status.json'
def write(**k):status.write_text(json.dumps({'pid':os.getpid(),**k},indent=2))
write(state='waiting',dependency=str(dependency))
while True:
    if dependency.exists():
        try:d=json.loads(dependency.read_text())
        except json.JSONDecodeError:time.sleep(5);continue
        if d['state'] in ('completed','failed','blocked_by_failed_dependency'):break
    time.sleep(15)
env=dict(os.environ);env.update(FINANCERL_EXECUTION='selective_fp32',FINANCERL_BASE_MODEL=str(ROOT/'local_models/InternVL3-2B'),
    HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
jobs=[(123,False),(456,False),(42,True)]
for seed,no_lora in jobs:
    name=f'strategy_ppo_{"no_lora" if no_lora else "internvl"}_seed{seed}_128'
    args=['strategy_ppo.py','--output',f'../campaign_runs/{name}','--seed',str(seed),'--iterations','128','--eval-every','32']
    if no_lora:args.append('--no-lora-train')
    write(state='running',job=name)
    result=subprocess.run([sys.executable,'-u',str(ROOT/'record_launch.py'),'--name',name,'--source','rl_strategy_trial','--',*args],
        cwd=ROOT,env=env,creationflags=subprocess.CREATE_NO_WINDOW)
    if result.returncode:write(state='failed',job=name,exit_code=result.returncode);sys.exit(result.returncode)
write(state='completed')
