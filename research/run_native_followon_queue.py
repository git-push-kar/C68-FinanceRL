"""Version-2 adaptive development continuation; no final test or deployment."""
import json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).parent
status=ROOT/'reports/native_followon_queue_status.json'
dependency=ROOT/'reports/readout_gate_queue_status.json'
results=[]
PILOT_GAIN_FLOOR=.01  # Prospective readout-pilot replication screen, not acceptance.
def read(path):return json.loads(path.read_text())
def write(**values):status.write_text(json.dumps({'pid':os.getpid(),'decision_version':2,
    'minimum_pilot_Sharpe_gain_for_replication':PILOT_GAIN_FLOOR,'results':results,**values},indent=2))
def stop(reason):
    write(state='needs_review',reason=reason)
    raise SystemExit(0)
write(state='waiting',dependency=str(dependency))
while True:
    try:state=read(dependency)['state']
    except (FileNotFoundError,json.JSONDecodeError):time.sleep(15);continue
    if state in ('completed','failed'):break
    time.sleep(15)
env=dict(os.environ)
env.update(FINANCERL_EXECUTION='selective_fp32',FINANCERL_BASE_MODEL=str(ROOT/'local_models/InternVL3-2B'),
    HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
def command(args):
    result=subprocess.run([sys.executable,'-u',*args],cwd=ROOT,env=env,creationflags=subprocess.CREATE_NO_WINDOW)
    if result.returncode:stop(f'Command failed with code {result.returncode}: {args}')
def record(name,source,args):
    write(state='running',job=name)
    command([str(ROOT/'record_launch.py'),'--name',name,'--source',source,'--',*args])
    results.append({'job':name,'exit_code':0})
def inspect(run_name):
    run=ROOT/'campaign_runs'/run_name
    try:
        run_status=read(run/'status.json');selection=read(run/'selection.json');audit=read(run/'reload_audit.json')
    except (FileNotFoundError,json.JSONDecodeError):stop('Completed, audited pilot is missing: '+run_name)
    if run_status['status']!='completed' or audit['status']!='passed':stop('Pilot technical verification failed: '+run_name)
    if not (run/'learning_comparison.json').exists():command(['compare_rl_learning.py','--run',str(run)])
    selected=selection['selected'];initial=selection['initial']
    delta=selected['development']['sharpe_proxy']-initial['development']['sharpe_proxy']
    positive=selected['step']>0 and delta>=PILOT_GAIN_FLOOR
    if positive and (audit['mean_sampled_Sharpe_gain'] is None or audit['mean_sampled_Sharpe_gain']<=0):
        stop('Expected-mixture gain disagrees with sampled-action mean; inspect execution mismatch: '+run_name)
    return positive,{'run':run_name,'selected_step':selected['step'],'development_Sharpe_gain':delta,
                     'sampled_mean_gain':audit['mean_sampled_Sharpe_gain']}
def train_and_audit(name,source,seed):
    record(name,source,['train_counterfactual.py','--output','../campaign_runs/'+name,
                       '--seed',str(seed),'--updates','64','--eval-every','16'])
    record('audit_'+name,source,['audit_text_run.py','--run','../campaign_runs/'+name])
def preserve():
    command(['collect_rl_evidence.py']);command(['backup_rl_progress.py'])

attention='sector_native_counterfactual_seed42_64'
positive,decision=inspect(attention)
if positive:
    source='rl_counterfactual_trial';stem='sector_native_counterfactual';report='native_attention_all_action_seed_comparison.json'
else:
    gate=ROOT/'checks/readout_lora_rl_gate/result.json'
    if not gate.exists() or read(gate)['status']!='passed':stop('Readout fallback lacks its own passed technical gate')
    source='rl_readout_trial';stem='sector_native_readout_counterfactual';report='native_readout_all_action_seed_comparison.json'
    write(state='running',decision='Attention pilot below replication screen; test declared readout configuration',pilot=decision)
    name=stem+'_seed42_64';train_and_audit(name,source,42)
    positive,decision=inspect(name);preserve()
    if not positive:
        write(state='completed_no_candidate',reason='Readout pilot did not meet replication screen; review before another method',pilot=decision)
        raise SystemExit(0)

write(state='running',decision='Replicate promising pilot; retain all outcomes',pilot=decision)
names=[stem+'_seed42_64']
for seed in (123,456):
    name=f'{stem}_seed{seed}_64';train_and_audit(name,source,seed)
    # Replication never stops or changes configuration because of financial loss.
    run=ROOT/'campaign_runs'/name
    if read(run/'reload_audit.json')['status']!='passed':stop('Replication audit failed: '+name)
    command(['compare_rl_learning.py','--run',str(run)]);names.append(name);preserve()
command(['compare_rl_seeds.py','--runs',*[str(ROOT/'campaign_runs'/n) for n in names],
         '--budget','64','--output',str(ROOT/'reports'/report)])
preserve();write(state='completed_development_only',runs=names,report=report,
                 note='Replication completed, not financial acceptance. No reserved outcomes accessed.')
