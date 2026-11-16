"""Bounded annual-refit diagnostic; no reserved outcomes and no GPU."""
import json, os, subprocess, sys, time
from pathlib import Path

R = Path(__file__).parent
status = R / 'reports/multiasset_annual_controls_status.json'
results = []
env = dict(os.environ)
env.update(CUDA_VISIBLE_DEVICES='-1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')

def read(p):
    return json.loads(p.read_text())

def write(**kw):
    status.write_text(json.dumps({'pid': os.getpid(), 'results': results, **kw}, indent=2))

def cmd(args):
    subprocess.run([sys.executable, '-u', *args], cwd=R, env=env,
                   check=True, creationflags=subprocess.CREATE_NO_WINDOW)

def launch(source, name, args):
    write(state='running', job=name)
    cmd(['record_launch.py', '--name', name, '--source', source, '--', *args])

try:
    write(state='waiting_for_risk_selector_CPU')
    while True:
        upstream = read(R / 'reports/risk_selector_controls_status.json')
        if upstream['state'] == 'completed_development_only':
            break
        if upstream['state'] == 'needs_review':
            raise RuntimeError('Upstream CPU diagnostic needs review')
        time.sleep(20)
    summaries = {}
    for arm in ('static', 'dynamic'):
        source = 'rl_multiasset_annual_' + arm
        parity = 'multiasset_annual_' + arm + '_parity'
        launch(source, parity, ['train_direct.py', '--output', '../checks/' + parity,
                               '--updates', '4', '--eval-every', '4'])
        for step in (0, 4):
            for split in ('validation', 'development'):
                new = read(R / 'checks' / parity / f'{split}_{step}.json')
                old = read(R / 'campaign_runs' / f'multiasset_{arm}_seed42_64' / f'{split}_{step}.json')
                if new != old:
                    raise AssertionError('Date parameterization changed original trajectory')
        launch(source, 'audit_' + parity, ['audit_numeric.py', '--run', '../checks/' + parity])
        (R / 'checks' / parity / 'date_parameterization_parity.json').write_text(
            json.dumps({'status': 'passed', 'steps': [0, 4], 'exact_full_evaluations': True}))
        summaries[arm] = []
        for year in (2022, 2023, 2024):
            dates = ['--train-start', f'{year-6}-01-01', '--train-end', f'{year-2}-12-31',
                     '--val-start', f'{year-1}-01-01', '--val-end', f'{year-1}-12-31',
                     '--dev-start', f'{year}-01-01', '--dev-end', f'{year}-12-31']
            names = []
            for seed in (42, 123, 456):
                name = f'multiasset_annual_{arm}_{year}_seed{seed}_64'
                names.append(name)
                launch(source, name, ['train_direct.py', '--output', '../campaign_runs/' + name,
                                     '--seed', str(seed), *dates])
                launch(source, 'audit_' + name, ['audit_numeric.py', '--run', '../campaign_runs/' + name])
                launch(source, 'controls_' + name, ['controls.py', '--run', '../campaign_runs/' + name])
                results.append(name)
            output = R / f'reports/multiasset_annual_{arm}_{year}_comparison64.json'
            cmd(['compare_rl_seeds.py', '--runs', *[str(R/'campaign_runs'/n) for n in names],
                 '--budget', '64', '--output', str(output)])
            summary = read(output)
            summaries[arm].append({'year': year, 'mean_Sharpe_gain': summary['mean_Sharpe_gain'],
                                   'positive_seeds': summary['positive_seeds'], 'source': str(output)})
    verdict = {}
    for arm, folds in summaries.items():
        verdict[arm] = {'folds': folds,
            'eligible_for_native_review': sum(f['mean_Sharpe_gain'] > 0 and f['positive_seeds'] >= 2 for f in folds) >= 2
            and all(f['mean_Sharpe_gain'] > -.1 for f in folds)}
    (R/'reports/multiasset_annual_summary.json').write_text(json.dumps({
        'scope': 'Reused development. Separate cash-reset annual folds, not a deployable stitched backtest.',
        'screen': 'At least two positive folds with at least two positive seeds each; no fold mean below -0.1. Eligibility is not automatic promotion or acceptance.',
        'arms': verdict}, indent=2))
    write(state='completed_development_only')
except Exception as exc:
    write(state='needs_review', reason=repr(exc))
    raise
