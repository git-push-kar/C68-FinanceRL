"""Describe the sealed final block without selecting or training any model."""
import argparse
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).parent


def sharpe(values):
    return np.sqrt(252)*values.mean(axis=-1)/np.maximum(values.std(axis=-1),1e-12)


def circular_indices(rng,n,block):
    starts=rng.integers(0,n,size=(n+block-1)//block)
    return ((starts[:,None]+np.arange(block))%n).ravel()[:n]


def summarize(folder,replicates=2000):
    folder=Path(folder);protocol=json.loads((folder/'frozen_protocol.json').read_text())
    status=json.loads((folder/'status.json').read_text())
    if status['status']!='completed':raise ValueError('Final evaluation is not complete')
    records={p.stem:json.loads(p.read_text()) for p in folder.glob('*.json')
             if p.stem not in {'frozen_protocol','summary','status','analysis'}}
    records={k:v for k,v in records.items() if 'daily' in v and 'metrics' in v}
    names=list(records);dates=[r['return_date'] for r in records['candidate_seed42']['daily']]
    for name,r in records.items():
        if [d['return_date'] for d in r['daily']]!=dates:raise ValueError('Unpaired dates: '+name)
    values=np.stack([[d['net_return'] for d in records[name]['daily']] for name in names])
    position={name:i for i,name in enumerate(names)}
    candidate=[position[f'candidate_seed{s}'] for s in (42,123,456)]
    comparators={
        'matched_initialization':[position[f'initial_seed{s}'] for s in (42,123,456)],
        'trained_no_lora':[position[f'trained_no_lora_seed{s}'] for s in (42,123,456)],
        'numeric_teacher_transformer':[position[f'numeric_attention_seed{s}_lr0003'] for s in (42,123,456)],
        'minimum_variance':[position['minimum_variance252_monthly']]*3,
        'HRP':[position['hrp252_monthly']]*3,
        'daily_equal_weight':[position['equal_weight_daily']]*3}
    if 'fixed_initial_risk252_monthly' in position:
        comparators['fixed_initial_risk']=[position['fixed_initial_risk252_monthly']]*3
    if all(f'warmup_seed{s}' in position for s in (42,123,456)):
        comparators['teacher_warmup']=[position[f'warmup_seed{s}'] for s in (42,123,456)]
    if all(f'numeric_utility_seed{s}' in position for s in (42,123,456)):
        comparators['numeric_direct_utility_trained_zero_lag']=[position[f'numeric_utility_seed{s}'] for s in (42,123,456)]
    if all(f'numeric_utility_lag1_seed{s}' in position for s in (42,123,456)):
        comparators['numeric_direct_utility_matched_lag']=[position[f'numeric_utility_lag1_seed{s}'] for s in (42,123,456)]
    if all(f'numeric_risk_seed{s}' in position for s in (42,123,456)):
        comparators['numeric_direct_risk']=[position[f'numeric_risk_seed{s}'] for s in (42,123,456)]
    if all(f'teacher_reference_seed{s}' in position for s in (42,123,456)):
        comparators['joint_teacher_reference']=[position[f'teacher_reference_seed{s}'] for s in (42,123,456)]
    pairs={label:(candidate,control) for label,control in comparators.items()}
    if 'secondary_lora_only_seed42' in position:
        pairs['secondary_adapter_only_vs_its_initialization']=(
            [position['secondary_lora_only_seed42']],[position['secondary_lora_only_initial_seed42']])
    point=sharpe(values);rng=np.random.default_rng(20261107);comparisons={}
    for label,(selected,control) in pairs.items():
        comparisons[label]={'mean_seed_delta_sharpe':float(np.mean(point[selected]-point[control])),
            'per_seed_delta_sharpe':(point[selected]-point[control]).tolist(),'optimization_replicates':len(selected),
            'block_bootstrap':{}}
    for block in (5,20,60):
        draws={label:[] for label in pairs}
        for _ in range(replicates):
            indices=circular_indices(rng,len(dates),block);stats=sharpe(values[:,indices])
            for label,(selected,control) in pairs.items():draws[label].append(float(np.mean(stats[selected]-stats[control])))
        for label,array in draws.items():
            comparisons[label]['block_bootstrap'][str(block)]={
                'percentile_95_interval':np.quantile(array,[.025,.975]).tolist(),
                'replicates':replicates}
    aggregate={key:float(np.mean([records[f'candidate_seed{s}']['metrics'][key] for s in (42,123,456)]))
        for key in ('return_total','return_annualized','sharpe_proxy','max_drawdown','mean_turnover','mean_trading_cost')}
    result={'protocol_sha256':protocol['protocol_sha256'],'dates':{'first':dates[0],'last':dates[-1],'count':len(dates)},
        'candidate_seed_mean':aggregate,'all_metrics':{name:r['metrics'] for name,r in records.items()},
        'allocation_quality':{name:r['allocation_quality_summary'] for name,r in records.items() if 'allocation_quality_summary' in r},
        'paired_comparisons':comparisons,
        'bootstrap_definition':'Circular blocks of shared date indices, paired across all methods; statistic is the mean seed-wise Sharpe difference (three primary optimization seeds; secondary adapter-only comparison has one). Resamples realized strategy returns, without retraining or re-running policies on synthetic prices. No ensemble portfolio is implied.',
        'limits':['One reserved post-publication period of about nine months, fixed historical universe, incomplete transferred data provenance.',
            'Risk-free rate zero; development architecture search was not preregistered.',
            'Percentile intervals depend on block/stationarity assumptions and do not prove equivalence.',
            'Intervals are individual comparisons, without a family-wise multiple-comparison correction.',
            'All models were frozen before these results; no final-period winner is selected.']}
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--results',required=True)
    p.add_argument('--output',required=True);args=p.parse_args();target=Path(args.output)
    if target.exists():raise FileExistsError(target)
    result=summarize(args.results);target.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({'output':str(target),'candidate_seed_mean':result['candidate_seed_mean'],
        'mean_delta_sharpe':{k:v['mean_seed_delta_sharpe'] for k,v in result['paired_comparisons'].items()}},indent=2))


if __name__=='__main__':main()
