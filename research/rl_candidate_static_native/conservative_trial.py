"""Compare a conservative starting policy with a short PPO-trained candidate."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import traceback

from college_run import ROOT, Runner, hardware_report, select_checkpoint, write_json
from collect_results import collect


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps', type=int, default=2048)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if args.steps < 512 or args.steps % 512:
        parser.error('Steps must be a positive multiple of 512, at least 512.')
    directory = ROOT / 'college_runs' / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') + '_conservative_trial')
    directory.mkdir(parents=True)
    runner = Runner(directory, args.dry_run)
    runner.environment.update(FINANCERL_BASE_MODEL=str(ROOT / 'local_models/InternVL3-2B'),
                              HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    status = {'status': 'running', 'role': 'development_trial', 'steps': args.steps,
              'base_model': 'OpenGVLab/InternVL3-2B', 'dtype': 'fp32'}
    write_json(directory / 'hardware.json', hardware_report())
    write_json(directory / 'trial_protocol.json', {
        'train_end': '2022-12-31', 'development_dates': ['2023-01-01', '2024-12-31'],
        'learning_rate': 1e-6, 'score_scale': .25, 'actor_init_scale': .01,
        'initial_log_std': -3., 'rebalance_rate': .1, 'seed': 42,
        'selection_rule': 'highest validation Sharpe, earlier checkpoint on ties',
        'comparison': 'selected trained checkpoint versus untrained initialization and daily equal weight',
        'note': 'These interventions are tested together; this trial does not identify each intervention separately. No final test data is accessed.'})
    code = 0
    try:
        runner.execute('regression_checks', ['-m', 'unittest', 'discover', '-s', 'tests', '-v'])
        training = directory / 'seed_42'
        runner.execute('train_conservative', ['train.py', '--data', 'data/nifty50_prices.csv',
            '--train-end', '2022-12-31', '--output', training, '--timesteps', args.steps,
            '--learning-rate', '0.000001', '--seed', '42', '--device', 'cuda', '--dtype', 'fp32',
            '--rollout-steps', '128', '--minibatch-size', '16', '--checkpoint-interval', '512',
            '--validation-interval', '512', '--validation-start', '2023-01-01', '--validation-end', '2024-12-31',
            '--score-scale', '.25', '--actor-init-scale', '.01', '--initial-log-std', '-3',
            '--rebalance-rate', '.1', '--evaluate-initial'])
        if args.dry_run:
            package = training / 'final'
        else:
            proof = json.loads((training / 'learning_check.json').read_text())
            if not proof.get('learning_verified') or proof.get('optimizer_updates', 0) < 1:
                raise RuntimeError('No verified learning updates')
            selection = select_checkpoint(training)
            write_json(training / 'selection.json', selection)
            package = Path(selection['selected']['package'])
        runner.execute('reload_and_validate', ['evaluate_policy.py', '--data', 'data/nifty50_prices.csv',
            '--artifacts', package, '--start', '2023-01-01', '--end', '2024-12-31', '--device', 'cuda',
            '--output', training / 'selected_validation.json'])
        if not args.dry_run:
            initial = json.loads((training / 'initial_validation.json').read_text())
            result = json.loads((training / 'selected_validation.json').read_text())
            learned, baseline, start = result['learned_policy'], result['daily_rebalanced_equal_weight'], initial['learned']
            comparison = {'role': 'development_only', 'untrained_initial_policy': start,
                          'selected_trained_policy': learned, 'daily_equal_weight': baseline,
                          'training_return_change_percentage_points': 100 * (learned['return_total'] - start['return_total']),
                          'training_sharpe_change': learned['sharpe_proxy'] - start['sharpe_proxy'],
                          'beats_initial_return_and_sharpe': learned['return_total'] > start['return_total'] and learned['sharpe_proxy'] > start['sharpe_proxy'],
                          'beats_equal_weight_return_and_sharpe': learned['return_total'] > baseline['return_total'] and learned['sharpe_proxy'] > baseline['sharpe_proxy'],
                          'note': 'These are validation comparisons after selection, not proof of generalization or LoRA-specific benefit.'}
            write_json(training / 'learning_value_comparison.json', comparison)
            print(json.dumps(comparison, indent=2), flush=True)
            status.update(comparison={key: comparison[key] for key in ('beats_initial_return_and_sharpe', 'beats_equal_weight_return_and_sharpe')})
        runner.execute('export_candidate', ['export_adapter.py', '--package', package,
            '--output', directory / 'packages/finance_conservative_candidate_seed42.zip'])
        status.update(status='dry_run_plan' if args.dry_run else 'completed',
                      interpretation='Candidate for evaluation; completion alone is not a quality pass.')
    except Exception as error:
        traceback.print_exc()
        (directory / 'error.txt').write_text(traceback.format_exc(), encoding='utf-8')
        status.update(status='failed', error=str(error))
        code = 1
    finally:
        write_json(directory / 'status.json', status)
        print('SEND BACK THIS ZIP:', collect(ROOT), flush=True)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
