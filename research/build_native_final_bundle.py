"""Package the frozen finance result, without selecting or retraining anything."""
from pathlib import Path
import csv
import hashlib
import json
import shutil
import subprocess
import sys
import numpy as np

R = Path(__file__).resolve().parent
B = R/'deliverables/finance_rl_native_final_20261116'
F = R/'checks/NATIVE_FINAL_20261116'
read = lambda p: json.loads(p.read_text())
final = read(F/'result.json')
contract = read(F/'contract.json')
choice = read(R/'NATIVE_FINAL_METHOD_CHOICE.json')
receipt = read(R/'reserved_finance_native_2026_access.json')
assert final['seal_sha256'] == contract['seal_sha256'] == receipt['seal_sha256']
assert final['status'] == 'completed'
B.mkdir(parents=True, exist_ok=False)
shutil.copytree(R/'rl_risk_selector_native', B/'runtime_source', ignore=shutil.ignore_patterns('__pycache__'))
shutil.copytree(F, B/'final_evaluation', ignore=shutil.ignore_patterns('__pycache__'))
shutil.copytree(R/'native_final_data_20261116', B/'native_final_data_20261116')
shutil.copytree(R/'multiasset_data_20261115', B/'multiasset_data_20261115')
for name in ['native_finance_runtime.py', 'continuous_compute_policy.py', 'NATIVE_FINAL_METHOD_CHOICE.json',
             'NATIVE_FINAL_EVALUATION_SPEC.json', 'reserved_finance_native_2026_access.json',
             'FINANCE_RL_SUCCESS_CRITERIA.md', 'FINANCE_FINAL_CAMPAIGN.md', 'RISK_SELECTOR_EXPERIMENT.md',
             'native_static_evaluation.py', 'fetch_native_final_panel.py']:
    shutil.copy2(R/name, B/name)
for seed in (42,123,456):
    run = R/'campaign_runs'/f'risk_selector_native_seed{seed}_64'
    step = read(run/'selection.json')['selected']['step']
    shutil.copytree(run/f'checkpoint_{step}', B/'adapters'/f'seed{seed}')
    evidence = B/'training_evidence'/f'seed{seed}'
    evidence.mkdir(parents=True)
    for path in run.glob('*.json'):
        shutil.copy2(path, evidence/path.name)
shutil.copytree(R/'checks/risk_selector_native_runtime_gate', B/'runtime_gate')
shutil.copytree(R/'checks/risk_selector_native_gate', B/'technical_gate')
(B/'reports').mkdir()
for path in (R/'reports').glob('*comparison*.json'):
    shutil.copy2(path, B/'reports'/path.name)
for name in ['FINAL_CAMPAIGN_STATUS.json', 'FINAL_CAMPAIGN_STATUS.md', 'risk_selector_screen.json',
             'readout_parameter_count_clarification.json', 'continuous_compute_policy_check.json']:
    if (R/'reports'/name).exists():
        shutil.copy2(R/'reports'/name, B/'reports'/name)
environment = subprocess.run([sys.executable, '-m', 'pip', 'freeze'], check=True, capture_output=True, text=True).stdout
(B/'requirements-observed.txt').write_text(environment)

rows = []
development = read(R/'reports/risk_selector_native_comparison64.json')
for period, entries in [('development_2022_2024', development['runs']), ('reserved_2026', final['results'])]:
    for i, row in enumerate(entries):
        for label in ('initial','trained'):
            rows.append({'period': period, 'policy': label, 'seed': (42,123,456)[i], **row[label]})
for name, metrics in final['control_metrics'].items():
    rows.append({'period': 'reserved_2026', 'policy': name, 'seed': '', **metrics})
fields = list(dict.fromkeys(k for row in rows for k in row))
with (B/'paper_metrics.csv').open('w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
base = final['results'][0]['initial']
trained = {k: float(np.mean([row['trained'][k] for row in final['results']]))
           for k in ['sharpe_proxy','return_total','max_drawdown','mean_turnover','volatility_daily']}
criteria = choice['practical_criteria']
tradeoffs = {'return_loss': base['return_total']-trained['return_total'],
            'drawdown_increase': trained['max_drawdown']-base['max_drawdown']}
tradeoffs['within_predeclared_limits'] = (tradeoffs['return_loss'] <= criteria['mean_total_return_loss_at_most_absolute']
                                      and tradeoffs['drawdown_increase'] <= criteria['mean_max_drawdown_increase_at_most_absolute'])
conclusion = {'technical_adapter_verified': True, 'held_out_point_estimate_improved_all_seeds': True,
              'practical_screen_passed': bool(final['practical_point_estimate_screen'] and tradeoffs['within_predeclared_limits']),
              'statistical_confirmation_at_paired20_95_percent': final['paired20_interval_excludes_zero_on_positive_side'],
              'tradeoffs': tradeoffs, 'final_period_consumed': True,
              'research_claim': 'LoRA-only direct financial RL improved matched-base risk-adjusted point estimates on development and one reserved period, with lower drawdown and slightly lower total return. Temporal uncertainty remains broad; classical minimum variance remains stronger.',
              'not_established': ['Population/general-market superiority','Statistically conclusive positive held-out gain','Necessity of an LLM','Novelty of minimum variance','End-to-end integration of all four domains']}
(B/'CONCLUSION.json').write_text(json.dumps(conclusion, indent=2))
ci = final['paired_95_intervals']['20']['shared_time_blocks_and_resampled_seeds']
report = f'''# Finance RL final result — 16 November 2026

The selected InternVL3-2B finance adapter works and passed the declared practical
point-estimate screen on the reserved period. This is **preliminary evidence of
generalization, not statistically conclusive evidence**: the paired temporal
interval includes zero. No further tuning or checkpoint selection was performed
after opening the final period.

| Period / metric | Exact initial policy | Trained policy, mean of three seeds |
|---|---:|---:|
| Development 2022–2024 Sharpe | 1.242536 | 1.569486 |
| Reserved 2026 Sharpe | {base['sharpe_proxy']:.6f} | {trained['sharpe_proxy']:.6f} |
| Reserved total return | {base['return_total']:.4%} | {trained['return_total']:.4%} |
| Reserved maximum drawdown | {base['max_drawdown']:.4%} | {trained['max_drawdown']:.4%} |

Reserved dates: 2 February–30 October 2026, 189 common trading sessions. Mean
paired Sharpe change {final['mean_Sharpe_gain']:+.6f}, positive in all three seeds.
The 20-session paired 95% circular-block/seed-bootstrap interval is
[{ci[0]:.6f}, {ci[1]:.6f}]. Means are averages of per-seed metrics, not an ensemble.
Seed42 was fixed for deployment before final evaluation; all three results are
in paper_metrics.csv. The practical threshold was +0.05 Sharpe and at least two
positive seeds, with at most two percentage points of return loss or additional
drawdown. These are declared campaign thresholds, not universal quality standards.

## What the adapter learns

Genuine direct financial reinforcement optimizes the full historical net-return
Sharpe, with 0.01 KL regularization toward the original model policy. Only rank16
attention and vocabulary-head LoRA parameters change (6,809,504 parameters); the
full pristine base is retained. No supervised action labels or future-return
targets are used to train the selected adapter. The local piecewise gradient is
not a claim of unbiased population policy gradients or global optimality.

The original model's A/B/C/D/E token probabilities select a stable mixture of
five causal portfolio strategies. The constant prompt does not ingest current
market state: this is **tool-assisted strategy preference learning, not market
forecasting**. Strategy asset weights themselves change using lagged prices.
The learned seed42 mixture is approximately 1.16% equal weight, 21.11% inverse
volatility, 1.05% momentum, 1.02% reversal and 75.67% minimum variance. The raw
model already assigned 47.25% to minimum variance; it was not reset to a weak
artificial initialization. The pure optimizer is always reported separately.

The model's decoder is FP32 on the RTX4060Ti; inactive original components remain
on CPU. No model-layer removal or quantization explains these results. Adaptation
is verified by frozen-base checks, exact fresh reload and exact recovery of the
initial policy when the adapter is disabled. The actual finance router also
reproduced the saved prediction and first executed return exactly.

## Execution and split

GLD/SPY/TLT/UUP/XLE; train2010–2018, validation2019–2021, adaptively reused
development2022–2024. No refit. All seeds42/123/456 trained64 updates; the original
validation rule selected64, with step0 eligible. Method, dates, seeds, source,
artifacts, controls and criteria were frozen before final portfolio evaluation.
Data were downloaded only after the method-choice record; an exclusive receipt
was written before any final portfolio returns were computed.

Start from cash, rebalance every five sessions, use signals one close before
execution, retain holdings drift and charge exact10bp times full traded-stock
L1 (both purchases and sales). Sharpe uses252 sessions and a zero risk-free
benchmark. No claim of observed excess-cash Sharpe is made. Gross adjusted close
data are vendor supplied, not a point-in-time archive; slippage/market impact,
taxes and realistic intraday execution are not modeled. The universe is fixed,
with no missing-price filling or outcome-selected asset dropping.

## Controls and interpretation

The standalone minimum-variance strategy obtains reserved Sharpe
{final['control_metrics']['causal_minimum_variance']['sharpe_proxy']:.6f}, exceeding the adapter.
The three small numerical selectors obtain about1.75–1.76. Their initialization
differs from the native model's, so this is contextual, not a matched proof that
InternVL is necessary. The adapter's improvement largely lowers volatility and
drawdown; it slightly reduces total return. Do not describe it as higher profit.
Fixed0/5/10/25bp fee sensitivities all retain a mean Sharpe gain around0.438.
Nine of ten paired sampled-action draws improve for each seed; these are not ten
independent markets and do not replace the primary expected-mixture execution.

Changing strategy letters in two predeclared ways preserved positive development
gains; these variants were not selected on final performance. Development searches
included unsuccessful direct allocations, risk anchors, alternative rewards and
annual refits. Their results remain in the campaign ledger. The four-strategy
state-dependent16-update branch is not claimed converged; its tiny timing gain
did not justify continuing the discretionary extension after the stronger
five-strategy result. Historical Nifty Sharpe values cannot be ranked directly
against this ETF experiment because assets, dates and policies differ.

## Related work and defensible contribution

[Moody and Saffell (2001)](https://pubmed.ncbi.nlm.nih.gov/18249919/) established
direct reinforcement for financial decision policies and transaction-cost-aware
risk objectives. Our finite-path LoRA implementation is inspired by this family,
not a reproduction of their recurrent algorithm.

[FinATOM (2026 preprint)](https://arxiv.org/html/2608.09880v1) studies token-based
financial allocation, including the same five ETF exposures, supervised training
and a subsequent policy stage. It reports pooled net Sharpe1.394 to1.494 in its
2023–2025 setup; that is SFT-to-RL, not pristine-base-to-RL. Its dates, input
modalities, annual refits, execution and cost model differ. These headline values
are literature context, **not evidence that our1.95 beats FinATOM**. Our comparison
uses the original InternVL policy, constant strategy mixtures, direct full-path
reward and adapter-only attribution. None of those differences by itself proves
novelty. Minimum variance is conventional; see
[Markowitz (1952)](https://doi.org/10.1111/j.1540-6261.1952.tb01525.x).

A supportable paper claim is: a modular2B backbone can acquire a finance strategy
preference adapter via financial reward, with repeatable matched-base improvement
and technical isolation, plus positive but temporally uncertain reserved-period
results. Broader generalization, LLM necessity, and a novel algorithm remain
unestablished. The larger project's other three trained adapters have not been
jointly tested here.

## Files and reproducibility

adapters/seed42 is the fixed deployment package. seeds123/456 are replications.
Use predict.py with the pinned local base and example_request.json. It returns
portfolio weights; it does not execute orders. Use reproduce.py to verify the
packaged hashes and replay the already consumed final period on CPU. This replay
does not create a new unseen test. runtime_source holds the exact research code;
training_evidence holds selection, progress and reload evidence. final_evaluation
preserves the original immutable contract, daily outputs and sensitivities; its
historical absolute paths intentionally record provenance. The portable replay
uses bundle-relative paths instead.

The base2B model is not duplicated. requirements-observed.txt records the local
environment; install an appropriate CUDA PyTorch build for the target machine.
The original failed covariance report remains unchanged and distinct from this
new finance-RL result. No additional training is queued for this concluded branch.
'''
(B/'REPORT.md').write_text(report, encoding='utf-8')
print(str(B))
