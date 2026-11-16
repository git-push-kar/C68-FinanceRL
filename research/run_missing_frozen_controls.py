"""Complete pre-existing Nifty PPO attribution controls, after GPU owner exits."""
import hashlib,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).parent;status=ROOT/'reports/missing_frozen_controls_status.json';results=[]
env=dict(os.environ)
env.update(FINANCERL_EXECUTION='selective_fp32',FINANCERL_BASE_MODEL=str(ROOT/'local_models/InternVL3-2B'),
 HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
def read(p):return json.loads(p.read_text())
def write(**kw):status.write_text(json.dumps({'pid':os.getpid(),'results':results,**kw},indent=2))
def command(args):
 r=subprocess.run([sys.executable,'-u',*args],cwd=ROOT,env=env,creationflags=subprocess.CREATE_NO_WINDOW)
 if r.returncode:raise RuntimeError(f'Exit {r.returncode}: {args}')
def source_check():
 expected=read(ROOT/'campaign_runs/strategy_ppo_internvl_seed123_128/protocol.json')['source_sha256']
 for name,sha in expected.items():
  if hashlib.sha256((ROOT/'rl_strategy_trial'/name).read_bytes()).hexdigest()!=sha:raise AssertionError('PPO source changed: '+name)
def launch(name,args):
 source_check();command(['record_launch.py','--name',name,'--source','rl_strategy_trial','--',*args])
def initialization(seed):
 names=[f'strategy_ppo_internvl_seed{seed}_128',f'strategy_ppo_no_lora_seed{seed}_128']
 def weights(name):
  base=ROOT/'campaign_runs'/name/'checkpoint_0'
  return {p.relative_to(base).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in base.rglob('*')
   if p.is_file() and p.suffix in ('.pt','.safetensors') and p.name!='training_state.pt'}
 a,b=map(weights,names)
 if len(a)<3 or a!=b:raise AssertionError('Initial module bytes differ')
 for split in ('validation','development'):
  x,y=[read(ROOT/'campaign_runs'/n/f'{split}_0.json') for n in names]
  if x!=y:raise AssertionError('Initial evaluation differs')
 (ROOT/f'reports/matched_full_frozen_lora_initialization_seed{seed}.json').write_text(json.dumps({'status':'passed','runs':names,'sha256':a,'identical_initial_evaluations':True},indent=2))
def main():
 source_check();write(state='waiting_for_readout_extended_queue')
 while True:
  upstream=read(ROOT/'reports/readout_extended_queue_status.json')
  if upstream['state']=='completed_development_only':break
  if upstream['state']=='needs_review':raise RuntimeError('GPU upstream needs review')
  time.sleep(20)
 for seed in (123,456):
  name=f'strategy_ppo_no_lora_seed{seed}_128';write(state='running',job=name)
  launch(name,['strategy_ppo.py','--output','../campaign_runs/'+name,'--seed',str(seed),'--iterations','128','--eval-every','32','--no-lora-train'])
  initialization(seed)
  launch('attribution_'+name,['audit_rl_attribution.py','--run','../campaign_runs/'+name,'--budget','128'])
  results.append(name);command(['backup_rl_progress.py'])
 command(['compare_full_frozen_ppo.py']);command(['backup_rl_progress.py'])
 write(state='completed_development_only',note='Matched 128-update, common 32-update selection grid; all seeds retained.')
try:main()
except Exception as exc:write(state='needs_review',reason=repr(exc));raise
