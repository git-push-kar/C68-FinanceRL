"""Three fixed-budget readout-LoRA replications; development only."""
import hashlib,json,os,subprocess,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).parent
status=ROOT/'reports/readout_extended_queue_status.json';results=[]
source=ROOT/'rl_readout_trial'
pilot=ROOT/'campaign_runs/sector_native_readout_counterfactual_seed42_64'
def read(path):return json.loads(path.read_text())
def write(**values):status.write_text(json.dumps({'pid':os.getpid(),'budget':256,'seeds':[42,123,456],'results':results,**values},indent=2))
def stop(reason):write(state='needs_review',reason=reason);raise SystemExit(1)
expected=read(pilot/'protocol.json')['source_sha256']
def check_source():
    actual={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in source.glob('*.py')}
    if actual!=expected:stop('Training source differs from audited pilot; review before proceeding')
env=dict(os.environ)
env.update(FINANCERL_EXECUTION='selective_fp32',FINANCERL_BASE_MODEL=str(ROOT/'local_models/InternVL3-2B'),
    HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
def command(args):
    result=subprocess.run([sys.executable,'-u',*args],cwd=ROOT,env=env,creationflags=subprocess.CREATE_NO_WINDOW)
    if result.returncode:stop(f'Command failed ({result.returncode}): {args}')
def record(name,args):
    check_source();write(state='running',job=name)
    command(['record_launch.py','--name',name,'--source','rl_readout_trial','--',*args])
    results.append({'job':name,'exit_code':0})
def prefix_check(run):
    checks=[]
    for step in (0,16,32,48,64):
        for split in ('validation','development'):
            a=read(pilot/f'{split}_{step}.json');b=read(run/f'{split}_{step}.json')
            error=float(np.max(abs(np.asarray(a['probabilities'])-np.asarray(b['probabilities']))))
            same=a['daily']==b['daily']
            if not np.isfinite(error) or error>1e-7 or not same:stop(f'Fresh prefix differs at {split} {step}: {error}')
            checks.append({'step':step,'split':split,'max_probability_error':error,'identical_daily_trajectory':same})
    (run/'pilot_prefix_audit.json').write_text(json.dumps({'status':'passed','checks':checks},indent=2))
write(state='starting');check_source()
names=[]
for seed in (42,123,456):
    name=f'sector_native_readout_counterfactual_seed{seed}_256';run=ROOT/'campaign_runs'/name
    record(name,['train_counterfactual.py','--output','../campaign_runs/'+name,'--seed',str(seed),
                 '--updates','256','--eval-every','16'])
    if seed==42:prefix_check(run)
    record('audit_'+name,['audit_text_run.py','--run','../campaign_runs/'+name])
    if read(run/'reload_audit.json')['status']!='passed':stop('Reload audit failed: '+name)
    command(['compare_rl_learning.py','--run',str(run)])
    command(['plot_rl_development.py','--run',str(run)])
    names.append(name);command(['collect_rl_evidence.py']);command(['backup_rl_progress.py'])
report=ROOT/'reports/native_readout_all_action_seed_comparison_budget256.json'
command(['compare_rl_seeds.py','--runs',*[str(ROOT/'campaign_runs'/n) for n in names],
         '--budget','256','--output',str(report)])
command(['backup_rl_progress.py']);write(state='completed_development_only',runs=names,report=str(report),
    note='All seeds retained. No reserved test or financial acceptance.')
