"""Summarize every completed selected experiment without choosing winners."""
from datetime import datetime
import json
from pathlib import Path
import re
import statistics
from execution_policy import IST

ROOT=Path(__file__).parent
rows=[]
for path in sorted((ROOT/'campaign_runs').glob('*/selection.json')):
    record=json.loads(path.read_text());selected=record.get('selected',{})
    if 'metrics' not in selected:continue
    initial=record.get('initial',{})
    metrics=selected['metrics'];before=initial.get('metrics',{})
    name=path.parent.name
    metadata=json.loads((path.parent/'run.json').read_text()) if (path.parent/'run.json').exists() else {}
    signal_lag=metadata.get('arguments',{}).get('signal_lag_closes',0)
    match=re.search(r'seed(\d+)',name)
    rows.append({'run':name,'seed':int(match.group(1)) if match else None,
        'signal_lag_closes':signal_lag,
        'selected_step':selected.get('step'),'teacher_KL':selected.get('fidelity',{}).get('teacher_KL'),
        'return_pct':100*metrics['return_total'],'sharpe':metrics['sharpe_proxy'],
        'drawdown_pct':100*metrics['max_drawdown'],
        'initial_return_pct':100*before['return_total'] if before else None,
        'initial_sharpe':before.get('sharpe_proxy'),
        'sharpe_gain':metrics['sharpe_proxy']-before['sharpe_proxy'] if before else None,
        'drawdown_change_pp':100*(metrics['max_drawdown']-before['max_drawdown']) if before else None})
groups={}
for prefix in ['covariance_teacher_seed','covariance_no_lora_seed','numeric_covariance_seed','numeric_attention_seed',
               'covariance_lora_only_seed','covariance_lora_mlp_seed','utility_seed','numeric_utility_seed',
               'covariance_risk_seed','covariance_risk_no_lora_seed','numeric_risk_seed','numeric_utility_lag1_seed',
               'covariance_risk_strict_seed','covariance_risk_strict_no_lora_seed']:
    selected=[r for r in rows if r['run'].startswith(prefix)
        and (prefix!='numeric_attention_seed' or r['run'].endswith('_lr0003'))]
    if selected:
        groups[prefix]={'completed_seeds':len(selected),'mean_sharpe':statistics.mean(r['sharpe'] for r in selected),
            'mean_return_pct':statistics.mean(r['return_pct'] for r in selected),
            'mean_sharpe_gain':statistics.mean(r['sharpe_gain'] for r in selected),
            'positive_seeds':sum(r['sharpe_gain']>0 for r in selected),
            'mean_drawdown_change_pp':statistics.mean(r['drawdown_change_pp'] for r in selected),
            'sharpe_seed_sample_sd':statistics.stdev(r['sharpe'] for r in selected) if len(selected)>1 else None}
output={'observed_ist':datetime.now(IST).isoformat(),'rows':rows,'groups':groups,
    'role':'reused 2023–2024 development; lag1-trained utility controls use one-close delay, other rows retain zero lag; no final-period results',
    'note':'Seed means describe optimization variation on the same market history, not independent markets.'}
(ROOT/'CAMPAIGN_RESULTS.json').write_text(json.dumps(output,indent=2))
lines=['# Recorded campaign results','',output['observed_ist'],'',output['role'],'',
    '| Run | Step | Signal delay | Return | Sharpe proxy | Max drawdown | ΔSharpe vs initialization |',
    '|---|---:|---:|---:|---:|---:|---:|']
for row in rows:
    gain=f"{row['sharpe_gain']:+.4f}" if row['sharpe_gain'] is not None else '—'
    lines.append(f"| {row['run']} | {row['selected_step']} | {row['signal_lag_closes']} | {row['return_pct']:.3f}% | {row['sharpe']:.4f} | {row['drawdown_pct']:.3f}% | {gain} |")
lines.extend(['','Selection criteria and all checkpoints remain in each run folder.',
    'Read LITERATURE_COMPARISON.md for published-method scope and classical controls.',
    'This table alone does not establish statistical superiority or equivalence.'])
(ROOT/'CAMPAIGN_RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps(output,indent=2))
