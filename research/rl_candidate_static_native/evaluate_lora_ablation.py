"""Compare a finance candidate with its own LoRA switched off at inference."""
import argparse
import json
from pathlib import Path

from env import FinanceRewardConfig, PortfolioEnv
from evaluate_policy import run_episode
from metrics import metrics_from_infos
from model import InternVLFinancePolicy, validate_package, write_json


def evaluate_ablation(policy, env, device):
    full = metrics_from_infos(run_episode(policy, env, device), env.initial_cash)
    layers = [m for m in policy.shared.model.modules() if hasattr(m, 'lora_A')]
    if not layers:
        raise ValueError('No LoRA layers found for this ablation')
    with policy.shared.model.disable_adapter():
        disabled = metrics_from_infos(run_episode(policy, env, device), env.initial_cash)
        if not all(layer.disable_adapters for layer in layers):
            raise RuntimeError('Finance execution reenabled adapters during the ablation')
    if any(layer.disable_adapters for layer in layers):
        raise RuntimeError('Adapters were not restored after ablation')
    return {'role': 'development_inference_ablation', 'with_finance_lora': full,
            'same_trained_heads_lora_disabled': disabled,
            'lora_return_difference_percentage_points': 100 * (full['return_total'] - disabled['return_total']),
            'note': 'This tests inference dependence on LoRA, not a separately trained no-LoRA control. A positive difference here is not proof of generalization.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True)
    parser.add_argument('--package', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    package, output = Path(args.package), Path(args.output)
    if output.exists():
        raise FileExistsError('Use a new ablation result path')
    manifest = validate_package(package)
    if manifest['run'].get('train_end', '9999') >= '2023-01-01':
        raise ValueError('Training overlaps development dates')
    cfg = json.loads((package / 'environment_config.json').read_text())
    env = PortfolioEnv(args.data, window=manifest['window'], reward_config=FinanceRewardConfig(**cfg['reward']),
                       start_date='2023-01-01', end_date='2024-12-31',covariance_window=cfg.get('covariance_window',0))
    if env.tickers != manifest['tickers']:
        raise ValueError('Dataset ticker order differs')
    policy = InternVLFinancePolicy.load_artifacts(package, device=args.device)
    report = evaluate_ablation(policy, env, args.device)
    write_json(output, report)
    env.close()
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
