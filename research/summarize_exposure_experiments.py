"""Transparent CPU feasibility comparison; never labels a CPU result an adapter."""
import json
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
ROOT=Path(__file__).parent
ARMS=[('Monte Carlo policy gradient','sector_exposure_numeric',1024,'exposure_numeric_seed_comparison_budget1024.json'),
      ('Soft Monte Carlo policy fitting','sector_exposure_soft_numeric',1024,'exposure_soft_numeric_seed_comparison_budget1024.json'),
      ('Minibatch direct portfolio Sharpe','sector_exposure_direct_numeric',256,'exposure_direct_numeric_seed_comparison_budget256.json'),
      ('Full-history direct portfolio Sharpe','sector_exposure_global_numeric',64,'exposure_global_numeric_seed_comparison_budget64.json')]
def read(p):
    try:return json.loads(p.read_text())
    except (FileNotFoundError,json.JSONDecodeError):return None
lines=['# Equity-exposure CPU feasibility comparisons','',f'Generated {datetime.now(timezone.utc).isoformat()}.','',
       '**All rows are CPU controls, not InternVL adapters. All financial outcomes are reused development (2022-2024).**','',
       'Same stock/cash task and untrained uniform expected exposure (.625). Every row uses its own validation-selected checkpoint. '
       'Direct training changes the reward, horizon and execution used in training; update counts are not matched simulation/compute budgets. '
       'The soft-fitting comparison changes the actor loss but retains Monte Carlo collection.','']
for name,prefix,budget,aggregate_name in ARMS:
    lines+=['## '+name,'','| Seed | Status | Selected step | Base Sharpe | Trained Sharpe | Change | Return change (pp) | Drawdown change (pp) | Dynamic minus fixed Sharpe |',
           '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for seed in (42,123,456):
        run=ROOT/f'campaign_runs/{prefix}_seed{seed}_{budget}';status=read(run/'status.json') or {}
        result=read(run/'learning_comparison.json');adapt=read(run/'exposure_adaptation.json')
        if not result or status.get('status')!='completed':
            lines.append(f'| {seed} | pending | — | — | — | — | — | — | — |');continue
        dynamic=f'{adapt["dynamic_minus_constant_Sharpe"]:+.6f}' if adapt else 'pending'
        lines.append(f'| {seed} | completed | {result["selected_step"]} | {result["initial"]["sharpe_proxy"]:.6f} | '
          f'{result["trained"]["sharpe_proxy"]:.6f} | {result["delta_Sharpe"]:+.6f} | {result["return_change_pp"]:+.4f} | '
          f'{result["drawdown_change_pp"]:+.4f} | {dynamic} |')
    aggregate=read(ROOT/'reports'/aggregate_name)
    if aggregate:
        ci=aggregate['paired_95_intervals']['20']['shared_time_blocks_and_resampled_seeds']
        lines+=['',f'Mean paired Sharpe change {aggregate["mean_Sharpe_gain"]:+.6f}; '
          f'{aggregate["positive_seeds"]}/3 positive seeds. Joint seed/time 20-day block interval [{ci[0]:+.6f}, {ci[1]:+.6f}].']
    lines+=['']
controls=read(ROOT/'reports/exposure_declared_controls.json')
if controls:
    lines+=['## Declared causal controls','','| Control | Development Sharpe | Total return (%) | Drawdown (%) |','|---|---:|---:|---:|']
    for name,value in controls['splits']['development'].items():
        lines.append(f'| {name} | {value["sharpe_proxy"]:.6f} | {100*value["return_total"]:.4f} | {100*value["max_drawdown"]:.4f} |')
lines+=['','Positive drawdown change means worse drawdown. Dynamic-minus-fixed compares against the average '
    'exposure derived from validation probabilities, not the best hindsight development exposure. '
    'All methods pay identical exact self-financing stock-notional fees. Block intervals do not correct adaptive development search. '
    'Sampled-action sensitivities are saved separately and share one market history. No reserved test has been accessed.','']
out=ROOT/'reports/EXPOSURE_CPU_COMPARISON.md';out.write_text('\n'.join(lines),encoding='utf-8');print(out)
