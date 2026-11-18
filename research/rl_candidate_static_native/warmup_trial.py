"""Warm up finance representations, then run PPO only after a development gate."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import traceback

from college_run import ROOT, Runner, hardware_report, select_checkpoint, write_json
from collect_results import collect


def warmup_gate(initial, candidate):
    before, after = initial['learned'], candidate['learned']
    correlation = candidate['internal_holdout_probe']['mean_cross_sectional_rank_correlation']
    return {'passed': bool(after['return_total'] - before['return_total'] > .001
                and (after.get('sharpe_proxy') or 0) - (before.get('sharpe_proxy') or 0) > .01
                and correlation is not None and correlation > .01),
            'return_change_percentage_points': 100 * (after['return_total'] - before['return_total']),
            'internal_holdout_rank_correlation': correlation,
            'rule': 'Gain over initialization exceeds 0.1 percentage point return and 0.01 Sharpe; internal holdout rank correlation exceeds 0.01.',
            'note': 'Predeclared development gate, not a significance test or final test.'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()
    output = ROOT / 'college_runs' / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') + '_feature_driven_warmup_then_ppo')
    output.mkdir(parents=True)
    runner = Runner(output, args.dry_run)
    runner.environment.update(FINANCERL_BASE_MODEL=str(ROOT / 'local_models/InternVL3-2B'),
                              HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    write_json(output / 'hardware.json', hardware_report())
    status = {'status': 'running', 'base_model': 'OpenGVLab/InternVL3-2B', 'role': 'development_experiment'}
    code = 0
    try:
        runner.execute('regression_checks', ['-m', 'unittest', 'discover', '-s', 'tests', '-v'])
        warm = output / 'warmup'
        runner.execute('supervised_warmup', ['warmup_finance.py', '--data', 'data/nifty50_prices.csv',
                       '--output', warm, '--steps', '512', '--batch-size', '4', '--device', 'cuda', '--feature-driven'])
        if args.dry_run:
            parent = warm / 'checkpoint_512'
            gate = {'passed': True, 'note': 'Dry-run plan only; real run applies the gate.'}
        else:
            selection = select_checkpoint(warm)
            write_json(warm / 'selection.json', selection)
            parent = Path(selection['selected']['package'])
            candidate = json.loads((warm / f"validation_{selection['selected']['steps']}.json").read_text())
            initial = json.loads((warm / 'initial_validation.json').read_text())
            gate = warmup_gate(initial, candidate)
        write_json(output / 'warmup_gate.json', gate)
        print('WARM-UP GATE:', json.dumps(gate), flush=True)
        if not gate['passed']:
            status.update(status='completed_warmup_only', interpretation='Warm-up did not pass the development gate. PPO was not started; no finished RL adapter is claimed.')
            return 0
        training = output / 'ppo_seed42'
        runner.execute('ppo_from_warmup', ['train.py', '--data', 'data/nifty50_prices.csv',
            '--train-end', '2022-12-31', '--output', training, '--initial-package', parent,
            '--timesteps', '2048', '--learning-rate', '.000001', '--seed', '42', '--device', 'cuda', '--dtype', 'fp32',
            '--rollout-steps', '128', '--minibatch-size', '16', '--checkpoint-interval', '512',
            '--validation-interval', '512', '--validation-start', '2023-01-01', '--validation-end', '2024-12-31',
            '--rebalance-rate', '.1', '--reward-mode', 'relative_log', '--reward-scale', '100', '--evaluate-initial'])
        if args.dry_run:
            package = training / 'final'
        else:
            selection = select_checkpoint(training)
            write_json(training / 'selection.json', selection)
            package = Path(selection['selected']['package'])
            before = json.loads((training / 'initial_validation.json').read_text())['learned']
            after = json.loads((training / f"validation_{selection['selected']['steps']}.json").read_text())['learned']
            write_json(training / 'ppo_increment.json', {'before_ppo': before, 'after_ppo': after,
                'return_change_percentage_points': 100 * (after['return_total'] - before['return_total']),
                'note': 'Warm-up benefit and additional PPO benefit must be reported separately.'})
        runner.execute('reload_and_validate', ['evaluate_policy.py', '--data', 'data/nifty50_prices.csv',
            '--artifacts', package, '--start', '2023-01-01', '--end', '2024-12-31', '--device', 'cuda',
            '--output', training / 'selected_validation.json'])
        runner.execute('lora_ablation', ['evaluate_lora_ablation.py', '--data', 'data/nifty50_prices.csv',
            '--package', package, '--device', 'cuda', '--output', training / 'lora_ablation.json'])
        runner.execute('export_candidate', ['export_adapter.py', '--package', package,
            '--output', output / 'packages/finance_warmup_ppo_candidate.zip'])
        status.update(status='dry_run_plan' if args.dry_run else 'completed',
                      interpretation='Development candidate. Review warm-up and PPO effects separately; no final test or real four-domain integration is claimed.')
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
