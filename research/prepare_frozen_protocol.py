"""Seal chosen models/controls before any reserved-period performance is read.

This script hashes files and reads metadata only. It never evaluates a model.
"""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).parent.resolve()


def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def relative(path):return Path(path).resolve().relative_to(ROOT).as_posix()


def selected_package(folder):
    selection=json.loads((folder/'selection.json').read_text())
    package=Path(selection['selected']['package'])
    if not package.is_absolute():package=ROOT/'covariance_trial'/package
    return package.resolve()


def neural_entry(name,package,selection=None):
    manifest=json.loads((package/'manifest.json').read_text())
    if manifest['run'].get('train_end','9999')>='2025-01-01':
        raise ValueError('Missing or overlapping training cutoff: '+name)
    if manifest.get('architecture_test_only'):raise ValueError('Random architecture control cannot be primary finance model')
    if name.startswith('candidate_') and (not manifest['run'].get('learning_verified') or manifest['run'].get('optimizer_updates',0)<1):
        raise ValueError('Primary candidate has no verified LoRA optimizer updates')
    if name.startswith(('candidate_','trained_no_lora_')):
        if manifest['run'].get('numerical_backend')!={'float32_matmul_precision':'highest',
                'cuda_matmul_allow_tf32':False,'cudnn_allow_tf32':False}:
            raise ValueError('Primary/matched package lacks strict training-precision evidence')
    for filename,expected in manifest['sha256'].items():
        if sha(package/filename)!=expected:raise ValueError('Changed package file: '+filename)
    entry={'name':name,'kind':'internvl','package':relative(package),
        'manifest_sha256':sha(package/'manifest.json'),'train_end':manifest['run']['train_end'],
        'training_numerical_backend':manifest['run'].get('numerical_backend','legacy or not recorded; strict final inference')}
    if selection is not None:
        entry.update(selection_file=relative(selection),selection_sha256=sha(selection))
    return entry


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--family',choices=['covariance_teacher_seed','covariance_lora_only_seed','covariance_lora_mlp_seed','covariance_risk_seed','covariance_risk_strict_seed','utility_seed'],required=True)
    p.add_argument('--output',required=True)
    args=p.parse_args();output=Path(args.output).resolve()
    method_file=ROOT/'METHOD_CHOICE.json';method=json.loads(method_file.read_text())
    if args.family!=method['selected_family']:raise ValueError('Family differs from the recorded pre-control method choice')
    if output.exists():raise FileExistsError(output)
    if (ROOT/'reserved_finance_2025_test.lock.json').exists():
        raise RuntimeError('Reserved period already locked: no new protocol after testing')
    observed=subprocess.run([sys.executable,'-m','pip','freeze'],capture_output=True,text=True,check=True).stdout
    expected=(ROOT/'ENVIRONMENT_LOCK_20261107.txt').read_text(encoding='utf-8-sig')
    if sorted(observed.splitlines())!=sorted(expected.splitlines()):
        raise RuntimeError('Actual Python packages differ from the environment lock')
    import torch
    torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    numerical={'float32_matmul_precision':torch.get_float32_matmul_precision(),
        'cuda_matmul_allow_tf32':torch.backends.cuda.matmul.allow_tf32,
        'cudnn_allow_tf32':torch.backends.cudnn.allow_tf32,'cudnn_benchmark':torch.backends.cudnn.benchmark,
        'deterministic_algorithms':torch.are_deterministic_algorithms_enabled()}
    entries=[]
    initial_family='covariance_teacher_seed' if args.family=='utility_seed' else args.family
    no_lora_family=('covariance_risk_strict_no_lora_seed' if args.family=='covariance_risk_strict_seed' else
        'covariance_risk_no_lora_seed' if args.family=='covariance_risk_seed' else 'covariance_no_lora_seed')
    for seed in (42,123,456):
        folder=ROOT/'campaign_runs'/f'{args.family}{seed}'
        if json.loads((folder/'selection.json').read_text())['selected']['step']!=method['selected_steps'][str(seed)]:
            raise ValueError('Primary checkpoint differs from the recorded method choice')
        entries.append(neural_entry(f'candidate_seed{seed}',selected_package(folder),folder/'selection.json'))
        entries.append(neural_entry(f'initial_seed{seed}',ROOT/'campaign_runs'/f'{initial_family}{seed}'/'initial'))
        if args.family!='covariance_teacher_seed':
            label='warmup' if args.family=='utility_seed' else 'teacher_reference'
            reference=ROOT/'campaign_runs'/f'covariance_teacher_seed{seed}'
            entries.append(neural_entry(f'{label}_seed{seed}',selected_package(reference),reference/'selection.json'))
        control_folder=ROOT/'campaign_runs'/f'{no_lora_family}{seed}'
        entries.append(neural_entry(f'trained_no_lora_seed{seed}',selected_package(control_folder),control_folder/'selection.json'))
        for attention in (False,True):
            name=f'numeric_attention_seed{seed}_lr0003' if attention else f'numeric_covariance_seed{seed}'
            control=ROOT/'campaign_runs'/name
            selection=json.loads((control/'selection.json').read_text())
            weights=control/f"checkpoint_{selection['selected']['step']}.pt"
            entries.append({'name':name,'kind':'numeric','attention':attention,'weights':relative(weights),
                'sha256':sha(weights),'selection_file':relative(control/'selection.json'),
                'selection_sha256':sha(control/'selection.json'),'train_end':'2022-12-31'})
        utility=ROOT/'campaign_runs'/f'numeric_utility_seed{seed}'
        utility_selection=json.loads((utility/'selection.json').read_text())
        weights=Path(utility_selection['selected']['weights'])
        entries.append({'name':f'numeric_utility_seed{seed}','kind':'numeric','attention':True,
            'weights':relative(weights),'sha256':sha(weights),
            'selection_file':relative(utility/'selection.json'),
            'selection_sha256':sha(utility/'selection.json'),'train_end':'2022-12-31'})
        lagged_utility=ROOT/'campaign_runs'/f'numeric_utility_lag1_seed{seed}'
        lagged_selection=json.loads((lagged_utility/'selection.json').read_text())
        weights=Path(lagged_selection['selected']['weights'])
        entries.append({'name':f'numeric_utility_lag1_seed{seed}','kind':'numeric','attention':True,
            'weights':relative(weights),'sha256':sha(weights),'training_signal_lag_closes':1,
            'selection_file':relative(lagged_utility/'selection.json'),
            'selection_sha256':sha(lagged_utility/'selection.json'),'train_end':'2022-12-31'})
        risk_control=ROOT/'campaign_runs'/f'numeric_risk_seed{seed}'
        risk_selection=json.loads((risk_control/'selection.json').read_text())
        weights=Path(risk_selection['selected']['weights'])
        entries.append({'name':f'numeric_risk_seed{seed}','kind':'numeric','attention':True,
            'weights':relative(weights),'sha256':sha(weights),
            'selection_file':relative(risk_control/'selection.json'),
            'selection_sha256':sha(risk_control/'selection.json'),'train_end':'2022-12-31'})
    attribution=ROOT/'campaign_runs/covariance_lora_only_seed42'
    entries.append(neural_entry('secondary_lora_only_seed42',selected_package(attribution),attribution/'selection.json'))
    entries.append(neural_entry('secondary_lora_only_initial_seed42',attribution/'initial'))
    classics=['equal_weight_daily','equal_weight_buy_hold','slow_uniform','inverse_vol20_slow',
              'minimum_variance252_monthly','hrp252_monthly','fixed_initial_risk252_monthly']
    publication_file=ROOT/'checks/BASE_PUBLICATION_METADATA.json'
    publication=json.loads(publication_file.read_text(encoding='utf-8-sig'))
    if publication['weight']['lfs']['oid']!='18b4cdbc8eab790f447feac717f4d0cecbb85caf8df198b3b2401c22506858e7':
        raise ValueError('Publication metadata does not describe the pinned weight file')
    protocol={'schema':1,'created_utc':datetime.now(timezone.utc).isoformat(),'selected_family':args.family,
        'test_start':'2025-05-01','test_end':'2026-01-30','cost_bps_times_stock_L1':10.,
        'method_choice_file':relative(method_file),'method_choice_sha256':sha(method_file),
        'method_decision_file':method['decision_record'],'method_decision_sha256':sha(ROOT/method['decision_record']),
        'deployment_seed':method['deployment_seed'],'deployment_selection_rule':method['deployment_selection_rule'],
        'signal_lag_closes':1,
        'execution_basis':'Full policy observation is available at the preceding trading close; execute at the next close with actual current holdings. Same delay for classical market-dependent controls. Portfolio starts in cash.',
        'publication_metadata_file':relative(publication_file),'publication_metadata_sha256':sha(publication_file),
        'chronology_basis':'Exact pinned weight object was uploaded 2025-04-10; official InternVL3 introduction 2025-04-11. Start May 2025, from cash, through the last available dataset date. Window changed before final-period performance was evaluated.',
        'release_source':'https://internvl.github.io/blog/2025-04-11-InternVL-3.0/',
        'data':'data/nifty50_prices.csv','dataset_sha256':sha(ROOT/'data/nifty50_prices.csv'),
        'entries':entries,'classical_controls':classics,
        'runtime_source_sha256':{p.name:sha(p) for p in (ROOT/'covariance_trial').glob('*.py')},
        'runtime_contract_sha256':{p.relative_to(ROOT/'covariance_trial').as_posix():sha(p)
            for p in (ROOT/'covariance_trial/contracts').rglob('*') if p.is_file()},
        'environment_lock_sha256':sha(ROOT/'ENVIRONMENT_LOCK_20261107.txt'),
        'numerical_backend':numerical,
        'interpretation':'First post-publication final-block evaluation in this campaign. No subsequent tuning or selecting models from these outcomes.',
        'history_limit':'No prior final-block result found in supplied records; inaccessible college-PC state cannot be certified.',
        'uncertainty':'Report paired return-series block bootstrap and all seed outcomes; seeds are not independent markets.',
        'reference_scope':'Classical methods adapted to identical panel/costs. Published headline numbers remain contextual.',
        'secondary_attribution':'One fixed seed with frozen numeric modules; compare to its own initialization. Not a final-period candidate for promotion or seed selection.',
        'pretraining_limit':'Adapter-training chronology is controlled; a complete point-in-time cutoff for pretrained InternVL data is not certified.'}
    canonical=json.dumps(protocol,sort_keys=True,separators=(',',':')).encode()
    protocol['protocol_sha256']=hashlib.sha256(canonical).hexdigest()
    output.mkdir(parents=True)
    (output/'protocol.json').write_text(json.dumps(protocol,indent=2))
    source=output/'runtime_source';source.mkdir()
    for name in protocol['runtime_source_sha256']:
        shutil.copy2(ROOT/'covariance_trial'/name,source/name)
    shutil.copytree(ROOT/'covariance_trial/contracts',source/'contracts')
    print(json.dumps({'protocol':str(output/'protocol.json'),'id':protocol['protocol_sha256'],
        'models':len(entries),'final_evaluation_performed':False},indent=2))


if __name__=='__main__':main()
