"""Audit completed pilots only after the existing single-GPU queue finishes."""
import json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).parent;dependency=ROOT/'reports/rl_replication_queue_status.json'
status=ROOT/'reports/rl_audit_queue_status.json'
def write(**k):status.write_text(json.dumps({'pid':os.getpid(),**k},indent=2))
write(state='waiting',dependency=str(dependency))
while True:
    try:d=json.loads(dependency.read_text())
    except (FileNotFoundError,json.JSONDecodeError):time.sleep(15);continue
    if d['state'] in ('completed','failed'):break
    time.sleep(15)
env=dict(os.environ);env.update(FINANCERL_EXECUTION='selective_fp32',FINANCERL_BASE_MODEL=str(ROOT/'local_models/InternVL3-2B'),
    HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
runs=['strategy_ppo_internvl_seed42_256_persistent','strategy_ppo_internvl_seed123_128',
    'strategy_ppo_internvl_seed456_128','strategy_ppo_no_lora_seed42_128']
jobs=[('audit_'+name,'rl_strategy_trial',['audit_rl_attribution.py','--run','../campaign_runs/'+name]) for name in runs]
jobs.append(('audit_sector_native_text_grpo_seed42_16','rl_text_trial',
    ['audit_text_run.py','--run','../campaign_runs/sector_native_text_grpo_seed42_16']))
results=[]
for name,source,args in jobs:
    target=ROOT/source/args[2]
    if not (target/'selection.json').exists():results.append({'job':name,'state':'skipped_missing_result'});continue
    write(state='running',job=name,results=results)
    r=subprocess.run([sys.executable,'-u',str(ROOT/'record_launch.py'),'--name',name,'--source',source,'--',*args],
        cwd=ROOT,env=env,creationflags=subprocess.CREATE_NO_WINDOW)
    results.append({'job':name,'exit_code':r.returncode})
write(state='completed' if all(r.get('exit_code')==0 for r in results) else 'completed_with_failures',results=results)
