import json,os,sys,subprocess,time
from pathlib import Path
import numpy as np
R=Path(__file__).parent;status=R/'reports/readout_only_ablation_status.json';results=[]
env=dict(os.environ);env.update(FINANCERL_EXECUTION='selective_fp32',FINANCERL_BASE_MODEL=str(R/'local_models/InternVL3-2B'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
def read(p):return json.loads(p.read_text())
def write(**kw):status.write_text(json.dumps({'pid':os.getpid(),'results':results,**kw},indent=2))
def cmd(args):subprocess.run([sys.executable,'-u',*args],cwd=R,env=env,check=True,creationflags=subprocess.CREATE_NO_WINDOW)
def launch(name,args):
 write(state='running',job=name);cmd(['record_launch.py','--name',name,'--source','rl_multiasset_readout_only','--',*args])
try:
 write(state='waiting_for_label_robustness_GPU')
 while True:
  upstream=read(R/'reports/label_robustness_status.json')
  if upstream['state']=='completed_development_only':break
  if upstream['state']=='needs_review':raise RuntimeError('Upstream GPU needs review')
  time.sleep(20)
 launch('multiasset_readout_only_gate',['text_gate.py'])
 gate=read(R/'checks/multiasset_readout_only_gate/result.json');fullgate=read(R/'checks/multiasset_dynamic_native_gate/result.json')
 if gate['status']!='passed' or not np.allclose(gate['initial_action_probabilities'],fullgate['initial_action_probabilities'],atol=1e-7,rtol=0):raise AssertionError('Readout-only initial gate differs')
 launch('multiasset_readout_only_smoke',['train_direct.py','--output','../checks/multiasset_readout_only_smoke','--updates','2','--eval-every','2'])
 launch('audit_multiasset_readout_only_smoke',['audit_text_run.py','--run','../checks/multiasset_readout_only_smoke'])
 for seed in (42,123,456):
  name=f'multiasset_readout_only_seed{seed}_16';r=R/'campaign_runs'/name
  launch(name,['train_direct.py','--output','../campaign_runs/'+name,'--seed',str(seed),'--updates','16','--eval-every','4'])
  for split in ('validation','development'):
   x=read(r/f'{split}_0.json');y=read(R/f'campaign_runs/multiasset_dynamic_native_seed{seed}_16/{split}_0.json')
   if x!=y:raise AssertionError('Initial financial trajectory differs from matched full-attention run')
  (r/'matched_initialization.json').write_text(json.dumps({'status':'passed','exact_initial_evaluations':True,'paired_full_attention_seed':seed}))
  launch('audit_'+name,['audit_text_run.py','--run','../campaign_runs/'+name]);results.append(name)
 cmd(['compare_rl_seeds.py','--runs',*[str(R/'campaign_runs'/n) for n in results],'--budget','16','--output',str(R/'reports/readout_only_ablation_comparison16.json')])
 write(state='completed_development_only')
except Exception as exc:write(state='needs_review',reason=repr(exc));raise
