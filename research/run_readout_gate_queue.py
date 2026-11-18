"""Technical fallback gate only; serialized after all existing GPU jobs."""
import json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).parent
dependency=ROOT/'reports/counterfactual_queue_status.json'
status=ROOT/'reports/readout_gate_queue_status.json'
def write(**values):status.write_text(json.dumps({'pid':os.getpid(),**values},indent=2))
write(state='waiting',dependency=str(dependency))
while True:
    try:state=json.loads(dependency.read_text())['state']
    except (FileNotFoundError,json.JSONDecodeError):time.sleep(15);continue
    if state in ('completed','failed'):break
    time.sleep(15)
env=dict(os.environ)
env.update(FINANCERL_EXECUTION='selective_fp32',FINANCERL_BASE_MODEL=str(ROOT/'local_models/InternVL3-2B'),
    HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
write(state='running',job='readout_lora_gate',financial_training=False)
result=subprocess.run([sys.executable,'-u',str(ROOT/'record_launch.py'),'--name','readout_lora_gate',
    '--source','rl_readout_trial','--','text_gate.py'],cwd=ROOT,env=env,creationflags=subprocess.CREATE_NO_WINDOW)
write(state='completed' if result.returncode==0 else 'failed',exit_code=result.returncode,financial_training=False)
sys.exit(result.returncode)
