import json,os,sys,time,subprocess
from pathlib import Path
R=Path(__file__).parent
env=dict(os.environ);env.update(FINANCERL_EXECUTION='selective_fp32',FINANCERL_BASE_MODEL=str(R/'local_models/InternVL3-2B'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
while True:
 s=json.loads((R/'reports/static_chronology_status.json').read_text())
 if s['state']=='completed_development_only':break
 if s['state']=='needs_review':raise RuntimeError('Training queue failed')
 time.sleep(20)
for fold in (1,2):
 for seed in (42,123,456):
  subprocess.run([sys.executable,'-u','audit_chronology_endpoint.py','--run',str(R/f'campaign_runs/nifty_static_f{fold}_seed{seed}_64')],cwd=R,env=env,check=True,creationflags=subprocess.CREATE_NO_WINDOW)
(R/'reports/chronology_endpoint_audits_status.json').write_text(json.dumps({'status':'passed','runs':6}))
