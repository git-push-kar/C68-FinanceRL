"""Conditional, matched 16-to-32 continuation. Never overlaps the existing GPU queue."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

R = Path(__file__).parent
status = R/'reports/native_matched_extension_status.json'
results = []
env = dict(os.environ)
env.update(FINANCERL_EXECUTION='selective_fp32', FINANCERL_BASE_MODEL=str(R/'local_models/InternVL3-2B'),
           HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
env.pop('CUDA_VISIBLE_DEVICES', None)

def read(p):
    return json.loads(p.read_text())

def write(**kw):
    status.write_text(json.dumps({'pid': os.getpid(), 'results': results, **kw}, indent=2))

def cmd(args, cpu=False):
    child_env = dict(env)
    if cpu:
        child_env['CUDA_VISIBLE_DEVICES'] = '-1'
    subprocess.run([sys.executable, '-u', *args], cwd=R, env=child_env,
                   check=True, creationflags=subprocess.CREATE_NO_WINDOW)

def launch(source, name, args):
    write(state='running', job=name)
    cmd(['record_launch.py', '--name', name, '--source', source, '--', *args])

def exact_continuation_check(actual, expected):
    from safetensors import safe_open
    for split in ('validation', 'development'):
        if read(actual/f'{split}_4.json') != read(expected/f'{split}_4.json'):
            raise AssertionError('Resumed step-four financial trajectory differs from uninterrupted training')
    path = Path('checkpoint_4/finance_lora/finance/adapter_model.safetensors')
    with safe_open(str(actual/path), framework='pt', device='cpu') as left:
        with safe_open(str(expected/path), framework='pt', device='cpu') as right:
            if set(left.keys()) != set(right.keys()):
                raise AssertionError('Resumed adapter keys differ')
            for key in left.keys():
                import torch
                if not torch.equal(left.get_tensor(key), right.get_tensor(key)):
                    raise AssertionError('Resume did not reproduce exact uninterrupted adapter tensors: '+key)
    (actual/'continuation_parity.json').write_text(json.dumps({
        'status': 'passed', 'exact_adapter_tensors': True,
        'exact_validation_and_development': True, 'reference': str(expected)}))

try:
    write(state='waiting_for_risk_label_robustness_GPU')
    while True:
        upstream = read(R/'reports/risk_label_robustness_status.json')
        if upstream['state'] in ('completed_development_only', 'completed_no_followup'):
            break
        if upstream['state'] == 'needs_review':
            raise RuntimeError('Risk label robustness upstream requires review')
        time.sleep(20)
    qualifying = []
    evidence = []
    for seed in (42, 123, 456):
        run = R/f'campaign_runs/multiasset_dynamic_native_seed{seed}_16'
        selected = read(run/'selection.json')['selected']['step']
        candidates = {c['step']: c for c in read(run/'candidates.json')}
        gain = candidates[16]['validation']['sharpe_proxy'] - candidates[12]['validation']['sharpe_proxy']
        qualifies = selected == 16 and gain >= .01
        if qualifies:
            qualifying.append(seed)
        evidence.append({'seed': seed, 'selected': selected, 'validation_gain_12_to_16': gain,
                         'qualifies': qualifies})
    (R/'reports/native_extension_trigger.json').write_text(json.dumps({
        'criterion': 'At least two full-state seeds select16 and improve validation Sharpe by at least0.01 from12 to16.',
        'development_used_for_trigger': False, 'evidence': evidence,
        'triggered': len(qualifying) >= 2}, indent=2))
    if len(qualifying) < 2:
        write(state='completed_no_extension', reason='Validation-only continuation trigger not met')
        sys.exit(0)
    arms = [
        ('full', 'rl_multiasset_dynamic_resume', 'multiasset_dynamic_native', 'multiasset_dynamic_native_smoke'),
        ('readout', 'rl_multiasset_readout_resume', 'multiasset_readout_only', 'multiasset_readout_only_smoke'),
    ]
    # Both continuation gates pass before either arm receives an extension.
    for arm, source, prefix, smoke in arms:
        name = f'multiasset_{arm}_resume_gate'
        launch(source, name, ['train_direct.py', '--resume', '../checks/'+smoke,
                             '--output', '../checks/'+name, '--updates', '4', '--eval-every', '4'])
        exact_continuation_check(R/'checks'/name, R/f'campaign_runs/{prefix}_seed42_16')
    for arm, source, prefix, _ in arms:
        names = []
        for seed in (42, 123, 456):
            name = f'{prefix}_seed{seed}_32_continued'
            names.append(name)
            launch(source, name, ['train_direct.py', '--resume', f'../campaign_runs/{prefix}_seed{seed}_16',
                                 '--output', '../campaign_runs/'+name, '--seed', str(seed),
                                 '--updates', '32', '--eval-every', '4'])
            launch(source, 'audit_'+name, ['audit_text_run.py', '--run', '../campaign_runs/'+name])
            cmd(['diagnose_multiasset_native.py', '--run', str(R/'campaign_runs'/name)], cpu=True)
            results.append(name)
        cmd(['compare_rl_seeds.py', '--runs', *[str(R/'campaign_runs'/name) for name in names],
             '--budget', '32', '--output', str(R/f'reports/native_{arm}_continued_comparison32.json')], cpu=True)
    write(state='completed_development_only')
except Exception as exc:
    write(state='needs_review', reason=repr(exc))
    raise
