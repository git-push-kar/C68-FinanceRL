import json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).parent
dependency=ROOT/'reports/rl_followup_queue_status.json'
status=ROOT/'reports/native_text_queue_status.json'
status.write_text(json.dumps({'state':'waiting','pid':os.getpid(),'dependency':str(dependency)},indent=2))
while True:
    if dependency.exists():
        try:d=json.loads(dependency.read_text())
        except json.JSONDecodeError:time.sleep(5);continue
        if d['state']=='completed':break
        if d['state'] in ('failed','blocked_by_failed_dependency'):
            status.write_text(json.dumps({'state':'blocked_by_failed_dependency','pid':os.getpid()},indent=2));sys.exit(1)
    time.sleep(15)
env=dict(os.environ);env.update(FINANCERL_EXECUTION='selective_fp32',FINANCERL_BASE_MODEL=str(ROOT/'local_models/InternVL3-2B'),
    HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
for name,args in [('native_text_rl_gate',['text_gate.py']),
    ('sector_native_text_grpo_seed42_16',['train_text_grpo.py','--output','../campaign_runs/sector_native_text_grpo_seed42_16',
                                       '--updates','16','--eval-every','8','--decisions','6'])]:
    status.write_text(json.dumps({'state':'running','job':name,'pid':os.getpid()},indent=2))
    result=subprocess.run([sys.executable,'-u',str(ROOT/'record_launch.py'),'--name',name,'--source','rl_text_trial','--',*args],
        cwd=ROOT,env=env,creationflags=subprocess.CREATE_NO_WINDOW)
    if result.returncode:
        status.write_text(json.dumps({'state':'failed','job':name,'exit_code':result.returncode},indent=2));sys.exit(result.returncode)
status.write_text(json.dumps({'state':'completed','pid':os.getpid()},indent=2))
