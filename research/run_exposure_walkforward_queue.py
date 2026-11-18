"""Declared historical CPU folds; no native model or reserved-test access."""
import json,os,subprocess,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).parent;status=ROOT/'reports/exposure_walkforward_queue_status.json';results=[]
env=dict(os.environ);env.update(CUDA_VISIBLE_DEVICES='-1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
def read(path):
    try:return json.loads(path.read_text())
    except (FileNotFoundError,json.JSONDecodeError):return {}
def write(**kw):status.write_text(json.dumps({'pid':os.getpid(),'device':'cpu','results':results,**kw},indent=2))
def command(args):
    r=subprocess.run([sys.executable,'-u',*args],cwd=ROOT,env=env,creationflags=subprocess.CREATE_NO_WINDOW)
    if r.returncode:raise RuntimeError(f'Exit {r.returncode}: {args}')
def launch(name,args):command(['record_launch.py','--name',name,'--source','rl_exposure_walkforward_trial','--',*args])
def main():
    write(state='waiting_for_exposure_global_numeric')
    while True:
        upstream=read(ROOT/'reports/exposure_global_numeric_queue_status.json')
        if upstream.get('state')=='completed_development_only':break
        if upstream.get('state')=='needs_review':raise RuntimeError('Upstream global experiment failed')
        time.sleep(20)
    # Correct the aggregate's legacy filename without altering its content.
    legacy=ROOT/'reports/exposure_global_numeric_seed_comparison_budget256.json'
    canonical=ROOT/'reports/exposure_global_numeric_seed_comparison_budget64.json'
    if not canonical.exists():
        if read(legacy).get('budget')!=64:raise AssertionError('Expected actual 64-update aggregate')
        canonical.write_bytes(legacy.read_bytes())
    write(state='checking_date_parameterization')
    launch('sector_exposure_walkforward_smoke',['train_direct.py','--output','../checks/exposure_walkforward_smoke','--updates','2','--eval-every','2'])
    launch('audit_sector_exposure_walkforward_smoke',['audit_numeric.py','--run','../checks/exposure_walkforward_smoke'])
    reference=ROOT/'checks/exposure_global_numeric_smoke';smoke=ROOT/'checks/exposure_walkforward_smoke'
    if read(smoke/'reload_audit.json').get('status')!='passed':raise AssertionError('Date-parameterization smoke failed')
    checks=[]
    for step in (0,2):
        for split in ('validation','development'):
            x=read(reference/f'{split}_{step}.json');y=read(smoke/f'{split}_{step}.json')
            error=float(np.max(abs(np.asarray(x['probabilities'])-np.asarray(y['probabilities']))))
            if error!=0 or x['daily']!=y['daily']:raise AssertionError('Date parameterization changed algorithm')
            checks.append({'step':step,'split':split,'maximum_probability_error':error,'identical_daily':True})
    (ROOT/'checks/exposure_walkforward_prefix_parity.json').write_text(json.dumps({'status':'passed','checks':checks},indent=2))
    for fold,train_end,val_start,val_end,dev_start,dev_end in [
        (1,'2009-12-31','2010-01-01','2012-12-31','2013-01-01','2015-12-31'),
        (2,'2012-12-31','2013-01-01','2015-12-31','2016-01-01','2018-12-31'),
        (3,'2015-12-31','2016-01-01','2018-12-31','2019-01-01','2021-12-31')]:
        names=[]
        for seed in (42,123,456):
            name=f'sector_exposure_walkforward_f{fold}_seed{seed}_64';run=ROOT/'campaign_runs'/name
            write(state='running',job=name)
            launch(name,['train_direct.py','--output','../campaign_runs/'+name,'--seed',str(seed),'--updates','64','--eval-every','4',
                '--train-end',train_end,'--validation-start',val_start,'--validation-end',val_end,'--development-start',dev_start,'--development-end',dev_end])
            launch('audit_'+name,['audit_numeric.py','--run','../campaign_runs/'+name])
            launch('adaptation_'+name,['controls.py','--run','../campaign_runs/'+name])
            command(['compare_rl_learning.py','--run',str(run)]);results.append(name);names.append(name)
        command(['compare_rl_seeds.py','--runs',*[str(ROOT/'campaign_runs'/n) for n in names],
            '--budget','64','--output',str(ROOT/f'reports/exposure_walkforward_f{fold}_seed_comparison_budget64.json')])
        command(['backup_rl_progress.py'])
    command(['summarize_exposure_walkforward.py']);command(['backup_rl_progress.py'])
    write(state='completed_development_only',note='All folds retained; historical method-search diagnostic, not a prospectively untouched test or native adapter.')
try:main()
except Exception as exc:write(state='needs_review',reason=repr(exc));raise
