"""Independent CPU-only calibration of the small-network PPO control."""
import json,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).parent;status=ROOT/'reports/numeric_control_queue_status.json'
env=dict(os.environ);env.update(CUDA_VISIBLE_DEVICES='-1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',
    HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
def write(**k):status.write_text(json.dumps({'pid':os.getpid(),'device':'CPU, CUDA hidden',**k},indent=2))
for label,lr in [('0p0003','0.0003'),('0p001','0.001')]:
    for seed in (42,123,456):
        name=f'strategy_ppo_numeric_lr{label}_seed{seed}_128';write(state='running',job=name)
        args=['strategy_ppo.py','--numeric','--policy-lr',lr,'--seed',str(seed),'--iterations','128','--eval-every','32',
            '--output','../campaign_runs/'+name]
        r=subprocess.run([sys.executable,'-u',str(ROOT/'record_launch.py'),'--name',name,'--source','rl_numeric_control','--',*args],
            cwd=ROOT,env=env,creationflags=subprocess.CREATE_NO_WINDOW)
        if r.returncode:write(state='failed',job=name,exit_code=r.returncode);sys.exit(r.returncode)
write(state='completed')
