import json,subprocess,sys,time,os
from pathlib import Path
R=Path(__file__).parent;env=dict(os.environ);env.update(CUDA_VISIBLE_DEVICES='-1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2');done=[]
for seed in (42,123,456):
 r=R/f'campaign_runs/multiasset_dynamic_native_seed{seed}_16'
 while True:
  if (r/'status.json').exists() and json.loads((r/'status.json').read_text()).get('status')=='completed':break
  parent=json.loads((R/'reports/multiasset_dynamic_native_status.json').read_text())
  if parent['state']=='needs_review':raise RuntimeError('GPU pipeline needs review')
  time.sleep(20)
 subprocess.run([sys.executable,'-u','diagnose_multiasset_native.py','--run',str(r)],cwd=R,env=env,check=True,creationflags=subprocess.CREATE_NO_WINDOW);done.append(seed)
 (R/'reports/native_diagnostics_status.json').write_text(json.dumps({'completed_seeds':done}))
