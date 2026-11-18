"""Readable evidence index; never promotes a development result to test success."""
from datetime import datetime,timezone
import json
from pathlib import Path
ROOT=Path(__file__).parent

def read(path):
    try:return json.loads(path.read_text())
    except (FileNotFoundError,json.JSONDecodeError):return None

def main():
    lines=['# Finance RL evidence index','',f'Generated {datetime.now(timezone.utc).isoformat()}.',
        '', '**Development evidence only. No new held-out 2026 result is claimed.**','',
        'Rows use the checkpoint selected by validation Sharpe, including initialization. '
        'Running rows are provisional. Datasets and periods differ between sections; '
        'their absolute Sharpe values cannot be ranked against one another.','']
    sections=[('Nifty49: reused 2023-2024 development, matched 128-iteration budget',[
        ('strategy_ppo_internvl_seed42_256_persistent',128),('strategy_ppo_internvl_seed123_128',128),
        ('strategy_ppo_internvl_seed456_128',128),('strategy_ppo_no_lora_seed42_128',128)]),
        ('Sector ETFs: reused 2022-2024 development, exploratory pilots',[
        ('sector_trajectory_grpo_numeric_seed42_64',64),('sector_trajectory_grpo_internvl_seed42_64',64),
        ('sector_native_text_grpo_seed42_16',16),('sector_native_counterfactual_seed42_64',64),
        ('sector_native_counterfactual_seed123_64',64),('sector_native_counterfactual_seed456_64',64),
        ('sector_native_readout_counterfactual_seed42_64',64),
        ('sector_native_readout_counterfactual_seed123_64',64),('sector_native_readout_counterfactual_seed456_64',64)]),
        ('Sector ETFs: readout-LoRA 256-update extension, reused development',[
        ('sector_native_readout_counterfactual_seed42_256',256),
        ('sector_native_readout_counterfactual_seed123_256',256),('sector_native_readout_counterfactual_seed456_256',256)]),
        ('Sector selector: calibrated prompt-information CPU control, 256 updates',[
        (f'sector_summary_numeric_lr0p001_seed{s}_256',256) for s in (42,123,456)]),
        ('Sector selector: stronger CPU control, unmatched 1024-update budget',[
        (f'sector_summary_numeric_lr0p001_seed{s}_1024',1024) for s in (42,123,456)]),
        ('Equity exposure: separate task, CPU feasibility only, not a native adapter',[
        (f'sector_exposure_numeric_seed{s}_1024',1024) for s in (42,123,456)]),
        ('Equity exposure: controlled soft actor-fitting CPU comparison',[
        (f'sector_exposure_soft_numeric_seed{s}_1024',1024) for s in (42,123,456)]),
        ('Equity exposure: direct reward CPU feasibility, distinct simulation budget',[
        (f'sector_exposure_direct_numeric_seed{s}_256',256) for s in (42,123,456)]),
        ('Equity exposure: full-history direct CPU feasibility, 64 full-path updates',[
        (f'sector_exposure_global_numeric_seed{s}_64',64) for s in (42,123,456)])]
    for title,entries in sections:
        lines.extend(['## '+title,'','| Run | Status | Selected / observed budget | Initial Sharpe | Trained Sharpe | Change |',
            '|---|---|---:|---:|---:|---:|'])
        for name,budget in entries:
            run=ROOT/'campaign_runs'/name;cs=read(run/'candidates.json')
            launch=read(ROOT/'launch_records'/name/'status.json') or {}
            status=read(run/'status.json') or launch
            if not cs:
                lines.append(f'| {name} | {status.get("status","not run")} | — | — | — | — |');continue
            eligible=[c for c in cs if c['step']<=budget]
            best=max(eligible,key=lambda c:(c['validation']['sharpe_proxy'],-c['step']))
            initial=eligible[0]['development']['sharpe_proxy'];trained=best['development']['sharpe_proxy']
            observed=max(c['step'] for c in eligible)
            state=status.get('status','running')
            if observed<budget or state!='completed':state+=' (provisional)'
            lines.append(f'| {name} | {state} | {best["step"]} / {observed} | {initial:.5f} | {trained:.5f} | {trained-initial:+.5f} |')
        lines.append('')
    lines.extend(['## What is still needed','',
        '- Verify saved reload, real adapter updates and frozen base.',
        '- Report all training seeds and frozen-LoRA/numeric controls.',
        '- Separate static strategy preference from state-dependent switching.',
        '- Evaluate sampled execution as well as the expected strategy mixture.',
        '- Freeze the final method, checkpoints and evaluation protocol before opening new test outcomes.',
        '- Package the selected adapter with its actual inference handler and execution contract.',''])
    control=read(ROOT/'reports/numeric_control_calibration.json')
    if control:
        lines.extend(['','## Small-network calibration','',
            f'CPU learning-rate grid: 0.0001, 0.0003, 0.001. Selected rate: {control["selected_lr"]}, by mean validation Sharpe across three seeds. All trials are in numeric_control_calibration.json.'])
    aggregate=read(ROOT/'reports/internvl_ppo_seed_comparison_budget128.json')
    if aggregate:
        ci=aggregate['paired_95_intervals']['20']['shared_time_blocks_and_resampled_seeds']
        lines.extend(['','## Matched PPO replication','',
            f'Mean paired Sharpe gain: {aggregate["mean_Sharpe_gain"]:+.5f}; positive seeds: {aggregate["positive_seeds"]}/{aggregate["training_seeds"]}. '
            f'20-day paired block interval with seed resampling: [{ci[0]:+.5f}, {ci[1]:+.5f}]. '
            'This does not establish a reliable gain or correct adaptive development search.'])
    lines.extend(['','## Adapter attribution at the matched PPO budget','',
        'Disabling LoRA after training retains the trained encoder/head. It is distinct from the separately trained frozen-LoRA control.',
        '', '| Run | Trained Sharpe | Disabled Sharpe | Difference | Mean sampled gain vs initial | Positive action draws |',
        '|---|---:|---:|---:|---:|---:|'])
    for name in ['strategy_ppo_internvl_seed42_256_persistent','strategy_ppo_internvl_seed123_128',
                 'strategy_ppo_internvl_seed456_128','strategy_ppo_no_lora_seed42_128']:
        audit=read(ROOT/'campaign_runs'/name/'attribution_budget_128/result.json')
        if audit:
            lines.append(f'| {name} | {audit["trained"]["sharpe_proxy"]:.5f} | {audit["adapter_disabled"]["sharpe_proxy"]:.5f} | '
                f'{audit["trained_minus_disabled_Sharpe"]:+.5f} | {audit["mean_sampled_Sharpe_gain"]:+.5f} | {audit["positive_sampled_gains"]}/10 |')
    lines.extend(['','Action draws share the same market history and are not independent market replications.',
                  '', '## Validation-selected static controls',''])
    for dataset in ['nifty','sector']:
        control=read(ROOT/f'reports/{dataset}_validation_static_grid.json')
        if control:
            lines.append(f'- {dataset}: weights {control["selected"]["weights"]}; development Sharpe {control["development"]["sharpe_proxy"]:.5f}. '
                'Selection uses the predeclared 35-point validation grid only; this is not RL.')
    lines.extend(['','## Execution queues',''])
    for name in ['rl_followup','native_text','rl_replication','rl_audit','counterfactual','readout_gate','native_followon','numeric_control','readout_extended','summary_numeric','exposure_numeric','exposure_soft_numeric','exposure_direct_numeric','exposure_global_numeric']:
        item=read(ROOT/f'reports/{name}_queue_status.json')
        if item:lines.append(f'- Queue `{name}`: {item["state"]}; job `{item.get("job","—")}`.')
    target=ROOT/'reports/RL_EVIDENCE_INDEX.md';target.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(target)

if __name__=='__main__':main()
