"""Matched three-seed finance benchmark, with staged risk warm-up and PPO."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import traceback

from college_run import ROOT, Runner, select_checkpoint, write_json
from collect_results import collect


def summarize(rows):
    improvements = [r['after']['sharpe_proxy'] - r['before']['sharpe_proxy'] for r in rows]
    return {'seeds': len(rows), 'mean_sharpe_improvement': statistics.mean(improvements),
            'sharpe_improvement_sample_sd': statistics.stdev(improvements) if len(rows) > 1 else None,
            'positive_seeds': sum(x > 0 for x in improvements),
            'mean_return_change_percentage_points': statistics.mean(100 * (r['after']['return_total'] - r['before']['return_total']) for r in rows),
            'mean_drawdown_change_percentage_points': statistics.mean(100 * (r['after']['max_drawdown'] - r['before']['max_drawdown']) for r in rows),
            'note': 'Seeds measure optimization variation on the same market period; they are not independent market histories.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    output = ROOT / 'college_runs' / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') + '_risk_benchmark')
    output.mkdir(parents=True)
    runner = Runner(output, args.dry_run)
    runner.environment.update(FINANCERL_BASE_MODEL=str(ROOT / 'local_models/InternVL3-2B'), HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    protocol = {'primary_metric': 'Sharpe proxy on net daily returns; no risk-free rate',
        'secondary_metrics': ['total return', 'max drawdown', 'turnover', 'costs'], 'seeds': [42, 123, 456],
        'baseline': 'Matched saved untrained model initialization, same architecture/encoder/execution/costs',
        'reference': 'Daily equal weight; raw returns are reported even when lower',
        'warmup_target': 'Causal inverse-volatility risk rule, not future-return forecasting',
        'ppo_reward': '100 * [(net - benchmark_net) - 10 * (net^2 - benchmark_net^2)]',
        'checkpoint_selection': 'Best 2023-2024 development Sharpe within each seed, earlier checkpoint on ties',
        'warmup_gate': 'Mean Sharpe improvement > 0.02, at least 2/3 positive seeds, mean drawdown worsening <= 0.5 percentage point, teacher probe correlation > 0.5 and teacher MSE improves >= 20% in each seed',
        'final_development_gate': 'Mean PPO-vs-untrained Sharpe improvement > 0.02, at least 2/3 positive seeds, mean drawdown worsening <= 0.5 percentage point',
        'reserved_test': '2025, evaluated once only after development gate; all seed models frozen before testing',
        'limitations': ['Current fixed asset universe; historical-membership bias is unresolved.', 'Development history was repeatedly inspected.', 'Three seeds are not a significance test.', 'No selection or retuning using reserved-test results.']}
    write_json(output / 'protocol.json', protocol)
    status, code = {'status': 'running', 'role': 'staged_development_benchmark'}, 0
    warmup_rows, parents, finals = [], {}, {}
    try:
        runner.execute('regression_checks', ['-m', 'unittest', 'discover', '-s', 'tests', '-v'])
        for seed in protocol['seeds']:
            warm = output / f'warmup_seed{seed}'
            runner.execute(f'warmup_{seed}', ['warmup_finance.py', '--data', 'data/nifty50_prices.csv',
                '--output', warm, '--steps', '512', '--batch-size', '4', '--device', 'cuda', '--seed', seed,
                '--feature-driven', '--target-kind', 'risk_rule'])
            if args.dry_run:
                parents[seed] = warm / 'checkpoint_512'
                continue
            selection = select_checkpoint(warm)
            write_json(warm / 'selection.json', selection)
            parents[seed] = Path(selection['selected']['package'])
            initial = json.loads((warm / 'initial_validation.json').read_text())
            chosen = json.loads((warm / f"validation_{selection['selected']['steps']}.json").read_text())
            warmup_rows.append({'seed': seed, 'before': initial['learned'], 'after': chosen['learned'],
                'initial_probe': initial['internal_holdout_probe'], 'after_probe': chosen['internal_holdout_probe']})
        if not args.dry_run:
            summary = summarize(warmup_rows)
            fidelity = all(r['after_probe']['mean_cross_sectional_rank_correlation'] is not None
                and r['after_probe']['mean_cross_sectional_rank_correlation'] > .5
                and r['after_probe']['rank_target_mse'] < .8 * r['initial_probe']['rank_target_mse'] for r in warmup_rows)
            passed = summary['mean_sharpe_improvement'] > .02 and summary['positive_seeds'] >= 2 and summary['mean_drawdown_change_percentage_points'] <= .5 and fidelity
            write_json(output / 'warmup_benchmark.json', {'rows': warmup_rows, 'summary': summary, 'passed': passed, 'teacher_fidelity_passed': fidelity})
            print('WARM-UP BENCHMARK:', json.dumps(summary), 'passed=', passed, flush=True)
            if not passed:
                status.update(status='completed_warmup_only', interpretation='Proper matched benchmark did not pass; PPO and reserved test not started.')
                return 0
        ppo_rows = []
        for seed in protocol['seeds']:
            training = output / f'ppo_seed{seed}'
            runner.execute(f'ppo_{seed}', ['train.py', '--data', 'data/nifty50_prices.csv', '--train-end', '2022-12-31',
                '--output', training, '--initial-package', parents[seed], '--timesteps', '2048', '--learning-rate', '.000001',
                '--seed', seed, '--device', 'cuda', '--dtype', 'fp32', '--rollout-steps', '128', '--minibatch-size', '16',
                '--checkpoint-interval', '512', '--validation-interval', '512', '--validation-start', '2023-01-01',
                '--validation-end', '2024-12-31', '--rebalance-rate', '.1', '--reward-mode', 'relative_mean_variance',
                '--reward-scale', '100', '--evaluate-initial'])
            if args.dry_run:
                finals[seed] = training / 'final'
            else:
                selection = select_checkpoint(training)
                write_json(training / 'selection.json', selection)
                finals[seed] = Path(selection['selected']['package'])
                chosen = json.loads((training / f"validation_{selection['selected']['steps']}.json").read_text())['learned']
                initial = warmup_rows[len(ppo_rows)]['before']
                before_ppo = json.loads((training / 'initial_validation.json').read_text())['learned']
                ppo_rows.append({'seed': seed, 'before': initial, 'after': chosen, 'after_warmup_before_ppo': before_ppo,
                    'ppo_only_sharpe_increment': chosen['sharpe_proxy'] - before_ppo['sharpe_proxy']})
            runner.execute(f'ablation_{seed}', ['evaluate_lora_ablation.py', '--data', 'data/nifty50_prices.csv',
                '--package', finals[seed], '--device', 'cuda', '--output', training / 'lora_ablation.json'])
            runner.execute(f'export_{seed}', ['export_adapter.py', '--package', finals[seed],
                '--output', output / 'packages' / f'finance_risk_ppo_seed{seed}.zip'])
        if not args.dry_run:
            summary = summarize(ppo_rows)
            passed = summary['mean_sharpe_improvement'] > .02 and summary['positive_seeds'] >= 2 and summary['mean_drawdown_change_percentage_points'] <= .5
            write_json(output / 'ppo_benchmark.json', {'rows': ppo_rows, 'summary': summary, 'passed': passed})
            status['development_benchmark_passed'] = passed
            if not passed:
                status.update(status='completed_development_only', interpretation='Candidate packages saved, but final benchmark gate failed; reserved test not accessed.')
                return 0
            lock = ROOT / 'reserved_finance_2025_test.lock.json'
            freeze = {'protocol': protocol, 'run': str(output), 'models': {str(seed): json.loads((finals[seed] / 'manifest.json').read_text()) for seed in protocol['seeds']}}
            try:
                with lock.open('x', encoding='utf-8') as stream:
                    json.dump(freeze, stream, indent=2)
            except FileExistsError:
                status.update(status='completed_development_only', interpretation='Reserved test already has a lock; not rerunning or overwriting it.')
                return 0
            write_json(output / 'frozen_test_protocol.json', freeze)
        for seed in protocol['seeds']:
            for name, package in [('untrained', output / f'warmup_seed{seed}/initial'), ('trained', finals[seed])]:
                runner.execute(f'test2025_{name}_{seed}', ['evaluate_policy.py', '--data', 'data/nifty50_prices.csv',
                    '--artifacts', package, '--start', '2025-01-01', '--end', '2025-12-31', '--device', 'cuda',
                    '--output', output / f'test2025_{name}_{seed}.json'])
        if not args.dry_run:
            test_rows = [{'seed': seed,
                'before': json.loads((output / f'test2025_untrained_{seed}.json').read_text())['learned_policy'],
                'after': json.loads((output / f'test2025_trained_{seed}.json').read_text())['learned_policy']} for seed in protocol['seeds']]
            write_json(output / 'reserved_test_benchmark.json', {'rows': test_rows, 'summary': summarize(test_rows),
                'note': 'Report this result honestly even if negative. The period is now used and cannot be treated as untouched in later tuning.'})
        status.update(status='dry_run_plan' if args.dry_run else 'completed', interpretation='Review development and reserved-test comparisons separately; real four-domain integration remains to be verified.')
    except Exception as error:
        traceback.print_exc()
        (output / 'error.txt').write_text(traceback.format_exc(), encoding='utf-8')
        status.update(status='failed', error=str(error)); code = 1
    finally:
        write_json(output / 'status.json', status)
        print('SEND BACK THIS ZIP:', collect(ROOT), flush=True)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
