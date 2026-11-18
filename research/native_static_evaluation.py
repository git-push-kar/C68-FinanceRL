"""Seal and evaluate verified constant-prompt native policies.

This does not select a method. A separate explicit specification selects it.
Development rehearsals cannot open 2025/2026. Final evaluation requires a
passed deployment-router gate and leaves an exclusive access receipt BEFORE
computing returns. This script never trains or changes checkpoint selection.
"""
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parent
SCHEMAS = {'candidate_static_text_v1', 'multiasset_static_text_v1',
           'risk_selector_static_text_v1'}
FINAL_RECEIPT = 'reserved_finance_native_2026_access.json'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def write_new(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False))
        stream.flush()
        os.fsync(stream.fileno())


def now():
    return datetime.now(timezone.utc).isoformat()


def constant_prompt(source):
    """Only accept a literal-string prompt; reject hidden input dependencies."""
    tree = ast.parse((source/'exposure_prompt.py').read_text())
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        raise ValueError('Unsupported prompt module')
    fn = tree.body[0]
    if fn.name != 'prompt' or fn.decorator_list or len(fn.body) != 1:
        raise ValueError('Only literal constant-prompt policies supported')
    ret = fn.body[0]
    if not isinstance(ret, ast.Return) or not isinstance(ret.value, ast.Constant) or not isinstance(ret.value.value, str):
        raise ValueError('Input-dependent prompt cannot use cached probabilities')
    return ret.value.value


def panel_metadata(data, start, end):
    """Reject data that the legacy loader would silently fill or drop."""
    import numpy as np
    import pandas as pd
    frame = pd.read_csv(data)
    if not {'Date', 'Ticker', 'Close'} <= set(frame):
        raise ValueError('Required price columns missing')
    frame['Date'] = pd.to_datetime(frame.Date, errors='raise')
    if frame.duplicated(['Date', 'Ticker']).any():
        raise ValueError('Duplicate asset/session rows')
    prices = frame.pivot(index='Date', columns='Ticker', values='Close').sort_index()
    if prices.isna().any().any() or not np.isfinite(prices.to_numpy()).all() or (prices <= 0).any().any():
        raise ValueError('Incomplete/nonpositive panel; no forward filling or asset dropping allowed')
    selected = prices.loc[start:end]
    if len(selected) < 60 or (prices.index < pd.Timestamp(start)).sum() < 254:
        raise ValueError('Insufficient evaluation sessions or causal warm-up')
    return {'tickers': prices.columns.tolist(), 'first': str(selected.index[0].date()),
            'last': str(selected.index[-1].date()), 'sessions': len(selected),
            'all_dates': [str(x.date()) for x in selected.index]}


def freeze(spec_file, output):
    import numpy as np
    spec = read(spec_file)
    purpose = spec['purpose']
    if purpose not in ('development_rehearsal', 'final_2026'):
        raise ValueError('Unknown purpose')
    start, end = spec['start'], spec['end']
    if start > end:
        raise ValueError('Invalid dates')
    if purpose == 'development_rehearsal' and end > '2024-12-31':
        raise ValueError('Rehearsals cannot open reserved dates')
    if purpose == 'final_2026':
        if start != '2026-02-02' or end != '2026-10-30':
            raise ValueError('Final date changes require an explicitly reviewed evaluator revision')
        if (ROOT/FINAL_RECEIPT).exists():
            raise RuntimeError('Final outcomes already accessed; cannot reseal another method')
        if not spec.get('method_choice_reason') or not spec.get('development_limitations'):
            raise ValueError('Record method choice and development limitations before testing')
    source, data = Path(spec['source']).resolve(), Path(spec['data']).resolve()
    if output.exists():
        raise FileExistsError(output)
    prompt = constant_prompt(source)
    panel = panel_metadata(data, start, end)
    gate = read(spec['technical_gate'])
    if gate['status'] != 'passed' or not gate['only_LoRA_trainable'] or gate['frozen_before'] != gate['frozen_after']:
        raise ValueError('Native adapter technical gate failed')
    minimum = spec['practical_min_mean_sharpe_gain']
    if not np.isfinite(minimum) or minimum <= 0:
        raise ValueError('Declare a positive practical-effect criterion before evaluation')
    entries, files, seeds = [], {}, []

    def track(path):
        path = Path(path).resolve()
        files[str(path)] = sha(path)
        return path

    track(spec_file)
    track(spec['technical_gate'])
    if spec.get('method_choice_file'):
        track(spec['method_choice_file'])
    for control in spec.get('secondary_fixed_mixtures', []):
        reference = read(track(control['evidence_file']))
        matrix = np.asarray(reference['probabilities'], dtype=float)
        if np.max(abs(matrix-matrix[0])) > 1e-12:
            raise ValueError('Secondary numerical policy is not constant')
        if not np.allclose(control['probabilities'], matrix[0], atol=1e-12, rtol=0):
            raise ValueError('Secondary mixture differs from its frozen validation evidence')
    if (ROOT/'continuous_compute_policy.py').exists():
        track(ROOT/'continuous_compute_policy.py')
    for run_name in spec['runs']:
        run = Path(run_name).resolve()
        protocol = read(track(run/'protocol.json'))
        seed = protocol['arguments']['seed']
        seeds.append(seed)
        if read(track(run/'status.json'))['status'] != 'completed':
            raise ValueError('Incomplete training run')
        audit = read(track(run/'reload_audit.json'))
        if audit['status'] != 'passed':
            raise ValueError('Fresh reload/adapter disable audit missing')
        for filename, expected in protocol['source_sha256'].items():
            if sha(source/filename) != expected:
                raise ValueError('Training source changed: '+filename)
        candidates = read(track(run/'candidates.json'))
        selected = read(track(run/'selection.json'))['selected']
        best = max(candidates, key=lambda x: (x['validation']['sharpe_proxy'], -x['step']))
        if selected != best or min(x['step'] for x in candidates) != 0:
            raise ValueError('Selection differs from validation Sharpe with zero eligible')
        if selected['step'] <= 0:
            raise ValueError('A zero-update selection cannot establish adapter improvement')
        if audit.get('selected_step') != selected['step']:
            raise ValueError('Reload audit is for a different checkpoint')
        pairs = {}
        for name, step in [('initial', 0), ('trained', selected['step'])]:
            package = run/f'checkpoint_{step}'
            if name == 'trained' and package.resolve() != Path(selected['package']).resolve():
                raise ValueError('Unexpected selected package location')
            manifest = read(track(package/'manifest.json'))
            for filename, expected in manifest['sha256'].items():
                path = (package/filename).resolve()
                if not path.is_relative_to(package.resolve()) or sha(track(path)) != expected:
                    raise ValueError('Adapter package integrity mismatch')
            record = read(package/'policy.json')
            if record['prompt_schema'] not in SCHEMAS or record['tickers'] != panel['tickers']:
                raise ValueError('Unsupported policy schema or changed universe')
            if record['base_revision'] != '899155015275a9b7338c7f4677e19c784e0e5a21':
                raise ValueError('Pristine model revision differs')
            probs = []
            for split in ('validation', 'development'):
                matrix = np.asarray(read(track(run/f'{split}_{step}.json'))['probabilities'], dtype=float)
                if matrix.ndim != 2 or not np.isfinite(matrix).all() or (matrix < 0).any() or np.max(abs(matrix.sum(1)-1)) > 1e-10:
                    raise ValueError('Invalid saved policy probabilities')
                if np.max(abs(matrix-matrix[0])) > 1e-12:
                    raise ValueError('Nonconstant policy; cached evaluation invalid')
                probs.append(matrix[0])
            if np.max(abs(probs[0]-probs[1])) > 1e-12:
                raise ValueError('Policy probabilities depend on evaluation inputs')
            pairs[name] = {'package': str(package), 'probabilities': probs[0].tolist(),
                           'schema': record['prompt_schema']}
        entries.append({'seed': seed, 'selected_step': selected['step'], 'run': str(run), **pairs})
    if sorted(seeds) != [42, 123, 456] or spec['deployment_seed'] not in seeds:
        raise ValueError('All three predeclared training seeds required')
    if purpose == 'final_2026':
        runtime = read(track(spec['runtime_gate']))
        deployment = next(x for x in entries if x['seed'] == spec['deployment_seed'])
        if runtime['status'] != 'passed' or Path(runtime['run']).resolve() != Path(deployment['run']) or runtime['selected_step'] != deployment['selected_step']:
            raise ValueError('Deployment adapter has no matching real-model router audit')
        if runtime['runtime_sha256'] != sha(ROOT/'native_finance_runtime.py'):
            raise ValueError('Audited deployment runtime changed')
        track(ROOT/'native_finance_runtime.py')
    output.mkdir(parents=True)
    snapshot = output/'runtime_source'
    snapshot.mkdir()
    for path in source.glob('*.py'):
        shutil.copy2(path, snapshot/path.name)
    if (source/'contracts').exists():
        shutil.copytree(source/'contracts', snapshot/'contracts')
    shutil.copy2(Path(__file__), output/'native_static_evaluation.py')
    snapshot_hashes = {str(p.relative_to(output)): sha(p) for p in output.rglob('*') if p.is_file()}
    contract = {'schema': 1, 'created_utc': now(), 'purpose': purpose,
                'spec': spec, 'panel': panel, 'entries': entries,
                'data': str(data), 'data_sha256': sha(data), 'source': str(source),
                'literal_prompt': prompt, 'tracked_files': files, 'snapshot_sha256': snapshot_hashes,
                'receipt_root': str(ROOT), 'fee_bps_times_full_stock_L1': 10,
                'signal_lag_closes': 1, 'rebalance_sessions': 5, 'starts_in_cash': True,
                'primary_execution': 'expected strategy mixture; constant probabilities, causal varying strategy weights',
                'analysis': {'risk_free_rate': 0, 'annualization': 252, 'bootstrap_replicates': 5000,
                             'bootstrap_seed': 20261107, 'blocks': [5, 20, 60],
                             'secondary_fee_bps': [0, 5, 10, 25],
                             'paired_action_seeds': list(range(10)),
                             'practical_min_mean_sharpe_gain': minimum,
                             'required_positive_seeds': 2,
                             'inference_rule': 'Point-estimate practical screen separate from paired 20-day interval excluding zero; neither guarantees economic generalization.'}}
    contract['seal_sha256'] = hashlib.sha256(canonical(contract)).hexdigest()
    write_new(output/'contract.json', contract)
    print(json.dumps({'sealed': str(output), 'purpose': purpose, 'outcomes_computed': False}))


def verify(directory):
    contract = read(directory/'contract.json')
    identity = contract.pop('seal_sha256')
    if hashlib.sha256(canonical(contract)).hexdigest() != identity:
        raise ValueError('Contract changed after sealing')
    contract['seal_sha256'] = identity
    for name, expected in contract['snapshot_sha256'].items():
        path = (directory/name).resolve()
        if not path.is_relative_to(directory.resolve()) or sha(path) != expected:
            raise ValueError('Frozen evaluation source changed')
    for name, expected in contract['tracked_files'].items():
        if sha(name) != expected:
            raise ValueError('Selected artifact/evidence changed: '+name)
    if sha(contract['data']) != contract['data_sha256']:
        raise ValueError('Evaluation data changed')
    if sha(__file__) != contract['snapshot_sha256']['native_static_evaluation.py']:
        raise ValueError('Execute the exact sealed evaluation code')
    return contract


def paired_analysis(pairs, contract):
    import numpy as np
    x = np.array([[r['net_return'] for r in a['daily']] for a, b in pairs])
    y = np.array([[r['net_return'] for r in b['daily']] for a, b in pairs])
    def sharpe(values):
        return np.sqrt(252)*values.mean(-1)/np.maximum(values.std(-1), 1e-12)
    gains = sharpe(y)-sharpe(x)
    n = x.shape[1]
    rng = np.random.default_rng(contract['analysis']['bootstrap_seed'])
    intervals = {}
    for block in contract['analysis']['blocks']:
        time_only, joint = [], []
        for _ in range(contract['analysis']['bootstrap_replicates']):
            starts = rng.integers(0, n, size=(n+block-1)//block)
            idx = ((starts[:, None]+np.arange(block)) % n).ravel()[:n]
            delta = sharpe(y[:, idx])-sharpe(x[:, idx])
            time_only.append(delta.mean())
            joint.append(delta[rng.integers(0, len(delta), size=len(delta))].mean())
        intervals[str(block)] = {'shared_time_blocks': np.quantile(time_only, [.025, .975]).tolist(),
                                  'shared_time_blocks_and_resampled_seeds': np.quantile(joint, [.025, .975]).tolist()}
    return {'mean_Sharpe_gain': float(gains.mean()), 'individual_Sharpe_gains': gains.tolist(),
            'positive_seeds': int((gains > 0).sum()), 'paired_95_intervals': intervals,
            'practical_point_estimate_screen': bool(gains.mean() >= contract['analysis']['practical_min_mean_sharpe_gain'] and (gains > 0).sum() >= 2),
            'paired20_interval_excludes_zero_on_positive_side': intervals['20']['shared_time_blocks_and_resampled_seeds'][0] > 0,
            'interpretation': 'Mean of paired per-seed Sharpe changes, not ensemble returns. Seeds and overlapping bootstrap samples are not independent market histories.'}


def evaluate_sealed(directory):
    contract = verify(directory)
    purpose = contract['purpose']
    receipt = (Path(contract['receipt_root'])/FINAL_RECEIPT if purpose == 'final_2026'
               else directory/'development_access.json')
    # Exclusive creation precedes importing financial code or constructing Market.
    # Failed/partial evaluation still counts as access and requires explicit recovery.
    write_new(receipt, {'access_started_utc': now(), 'seal_sha256': contract['seal_sha256'],
                        'contract': str(directory/'contract.json'), 'purpose': purpose})
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    os.environ['OMP_NUM_THREADS'] = '2'
    os.environ['MKL_NUM_THREADS'] = '2'
    sys.path.insert(0, str(directory/'runtime_source'))
    from candidate_market import Market, evaluate, NAMES
    from execution_policy import wait_for_compute_window
    import numpy as np
    import torch
    torch.set_num_threads(2)
    wait_for_compute_window()
    spec = contract['spec']
    market = Market(contract['data'], spec['start'], spec['end'])
    pairs, results = [], []
    def run(probs):
        result = evaluate(None, market, 'cpu', fixed=probs)
        if [x['return_date'] for x in result['daily']] != contract['panel']['all_dates']:
            raise ValueError('Evaluation omitted or changed frozen sessions')
        return result
    for entry in contract['entries']:
        initial = run(entry['initial']['probabilities'])
        trained = run(entry['trained']['probabilities'])
        pairs.append((initial, trained))
        result = {'seed': entry['seed'], 'selected_step': entry['selected_step'],
                  'initial': initial, 'trained': trained}
        write_new(directory/f'seed{entry["seed"]}.json', result)
        results.append({'seed': entry['seed'], 'selected_step': entry['selected_step'],
                        'initial': initial['metrics'], 'trained': trained['metrics']})
    controls = {name: run(np.eye(len(NAMES))[i]) for i, name in enumerate(NAMES)}
    for control in spec.get('secondary_fixed_mixtures', []):
        if control['name'] in controls:
            raise ValueError('Duplicate control name')
        controls[control['name']] = run(control['probabilities'])
    write_new(directory/'controls.json', controls)
    analysis = paired_analysis(pairs, contract)
    # Fixed secondary analyses: no new checkpoints or selection after outcomes.
    from functools import partial
    import candidate_market
    import exposure_market
    from cash_execution import execute
    from metrics import metrics_from_infos
    costs = {}
    for fee in contract['analysis']['secondary_fee_bps']:
        candidate_market.execute = partial(execute, fee_bps=fee)
        exposure_market.execute = partial(execute, fee_bps=fee)
        rows = []
        for entry in contract['entries']:
            a = run(entry['initial']['probabilities'])['metrics']
            b = run(entry['trained']['probabilities'])['metrics']
            rows.append({'seed': entry['seed'], 'initial': a, 'trained': b,
                         'Sharpe_gain': b['sharpe_proxy']-a['sharpe_proxy']})
        costs[str(fee)] = rows
    candidate_market.execute = exposure_market.execute = execute
    sampled = []
    for entry in contract['entries']:
        draws = []
        for seed in contract['analysis']['paired_action_seeds']:
            row = {'action_seed': seed}
            for name in ('initial', 'trained'):
                rng = np.random.default_rng(seed)
                probs = np.asarray(entry[name]['probabilities'])
                probs = probs/probs.sum()
                state = market.reset()
                while True:
                    wait_for_compute_window()
                    _, targets = state
                    action = min(int(np.searchsorted(np.cumsum(probs), rng.random())), len(probs)-1)
                    state, _, done = market.step(targets[action])
                    if done:
                        break
                row[name] = metrics_from_infos(market.infos)
            row['Sharpe_gain'] = row['trained']['sharpe_proxy']-row['initial']['sharpe_proxy']
            draws.append(row)
        sampled.append({'training_seed': entry['seed'], 'draws': draws,
                        'mean_sampled_Sharpe_gain': float(np.mean([x['Sharpe_gain'] for x in draws]))})
    write_new(directory/'secondary_sensitivity.json', {'fixed_fee_metrics': costs,
              'paired_sampled_execution': sampled,
              'scope': 'Prespecified sensitivities, no checkpoint selection. Action draws share the same market history; primary execution remains the expected mixture.'})
    report = {'status': 'completed', 'completed_utc': now(), 'purpose': purpose,
              'seal_sha256': contract['seal_sha256'], 'results': results, **analysis,
              'control_metrics': {name: value['metrics'] for name, value in controls.items()},
              'scope': ('Previously reused development; rehearsal only.' if purpose == 'development_rehearsal'
                        else 'Single reserved market period; no further selection or tuning on these outcomes.')}
    write_new(directory/'result.json', report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('freeze')
    p.add_argument('--spec', required=True)
    p.add_argument('--output', required=True)
    p = sub.add_parser('evaluate')
    p.add_argument('--sealed', required=True)
    args = parser.parse_args()
    if args.command == 'freeze':
        freeze(Path(args.spec).resolve(), Path(args.output).resolve())
    else:
        evaluate_sealed(Path(args.sealed).resolve())
