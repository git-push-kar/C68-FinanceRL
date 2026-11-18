import json,os,sys,subprocess,time
from pathlib import Path
R=Path(__file__).parent;results=[];status=R/'reports/risk_selector_controls_status.json'
env=dict(os.environ);env.update(CUDA_VISIBLE_DEVICES='-1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
def write(**kw):status.write_text(json.dumps({'pid':os.getpid(),'results':results,**kw},indent=2))
def cmd(args):subprocess.run([sys.executable,'-u',*args],cwd=R,env=env,check=True,creationflags=subprocess.CREATE_NO_WINDOW)
def launch(source,name,args):
 write(state='running',job=name);cmd(['record_launch.py','--name',name,'--source',source,'--',*args])
try:
 write(state='waiting_for_risk_reward_controls')
 while True:
  upstream=json.loads((R/'reports/asset_risk_reward_controls_status.json').read_text())
  if upstream['state']=='completed_development_only':break
  if upstream['state']=='needs_review':raise RuntimeError('Risk-reward control failed')
  time.sleep(20)
 for variant in ('static','dynamic'):
  source='rl_risk_selector_'+variant;names=[]
  smoke=f'risk_selector_{variant}_smoke'
  launch(source,smoke,['train_direct.py','--output','../checks/'+smoke,'--updates','2','--eval-every','2'])
  launch(source,'audit_'+smoke,['audit_numeric.py','--run','../checks/'+smoke])
  for seed in (42,123,456):
   name=f'risk_selector_{variant}_seed{seed}_64';names.append(name)
   launch(source,name,['train_direct.py','--output','../campaign_runs/'+name,'--seed',str(seed)])
   launch(source,'audit_'+name,['audit_numeric.py','--run','../campaign_runs/'+name])
   launch(source,'controls_'+name,['controls.py','--run','../campaign_runs/'+name]);results.append(name)
  cmd(['compare_rl_seeds.py','--runs',*[str(R/'campaign_runs'/n) for n in names],'--budget','64','--output',str(R/f'reports/risk_selector_{variant}_comparison64.json')])
 screen={}
 for arm in ('static','dynamic'):
  aggregate=json.loads((R/f'reports/risk_selector_{arm}_comparison64.json').read_text())
  timing=[json.loads((R/f'campaign_runs/risk_selector_{arm}_seed{seed}_64/candidate_controls.json').read_text())['dynamic_minus_constant_Sharpe'] for seed in (42,123,456)]
  base_screen=aggregate['mean_Sharpe_gain']>.05 and aggregate['positive_seeds']>=2
  screen[arm]={'mean_Sharpe_gain':aggregate['mean_Sharpe_gain'],'positive_seeds':aggregate['positive_seeds'],'timing_gains':timing,'eligible_for_native_review':base_screen and (arm=='static' or (sum(timing)/3>.01 and sum(x>0 for x in timing)>=2))}
 (R/'reports/risk_selector_screen.json').write_text(json.dumps({'scope':'Development screen only; optimizer-alone remains mandatory comparator','arms':screen},indent=2))
 write(state='completed_development_only')
except Exception as exc:write(state='needs_review',reason=repr(exc));raise
