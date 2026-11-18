"""Single precommitted post-publication evaluation, with an exclusive global lock.

No training or model selection occurs here. Resume can only finish the same
cryptographically frozen protocol and never overwrites completed results.
"""
import argparse
from datetime import datetime,timezone
import gc
import hashlib
import json
from pathlib import Path
import sys
import traceback
import subprocess

import psutil
import numpy as np
import torch

from classical_panel import episode,min_variance
from env import PortfolioEnv,FinanceRewardConfig
from evaluate_policy import run_episode
from evaluate_sensitivity import lagged_episode
from execution_policy import wait_for_compute_window,configure_finance_precision
from metrics import metrics_from_infos
from model import InternVLFinancePolicy,validate_package,write_json
from numeric_covariance_control import NumericCovariancePolicy
from shared_backbone import SharedBackbone
from static_risk_control import episode as static_risk_episode

ROOT=Path(__file__).resolve().parent.parent


def allocation_observer(rows,signal_lag=0):
    def observe(env,observation,scores):
        if (env.index-env.start_index)%env.reward_config.rebalance_interval:return
        cov=observation['covariance'].astype(np.float64)
        weights=env._target_weights(scores);reference=min_variance(cov,cap=env.reward_config.max_weight)
        variance=float(weights@cov@weights);optimum=float(reference@cov@reference)
        rows.append({'decision_date':str(env.dates[env.index])[:10],
            'covariance_signal_close_date':str(env.dates[env.index-signal_lag])[:10],
            'variance_to_numerical_optimizer':variance/optimum,'volatility_to_numerical_optimizer':float(np.sqrt(variance/optimum)),
            'variance_to_uniform':variance/float(cov.mean()),'allocation_L1_to_optimizer':float(np.abs(weights-reference).sum()),
            'target_weights':weights.tolist(),'optimizer_weights':reference.tolist()})
    return observe


def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def checked_path(relative):
    path=(ROOT/relative).resolve()
    if not path.is_relative_to(ROOT):raise ValueError('Artifact escapes campaign root')
    return path


def verify(protocol):
    content={k:v for k,v in protocol.items() if k!='protocol_sha256'}
    actual=hashlib.sha256(json.dumps(content,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    if actual!=protocol['protocol_sha256']:raise ValueError('Protocol hash mismatch')
    if protocol.get('schema')!=1:raise ValueError('Unknown frozen protocol schema')
    if (protocol['test_start'],protocol['test_end'])!=('2025-05-01','2026-01-30'):
        raise ValueError('This runner is exclusively for the declared final block')
    required={f'{kind}_seed{seed}' for kind in ('candidate','initial','trained_no_lora') for seed in (42,123,456)}
    names=[entry['name'] for entry in protocol['entries']]
    if not required.issubset(names) or len(names)!=len(set(names)):
        raise ValueError('Missing matched frozen candidates/controls or duplicate result names')
    core={'run_frozen_test.py','model.py','config.py','env.py','allocation.py','shared_backbone.py',
          'execution_policy.py','evaluate_policy.py','classical_panel.py','numeric_covariance_control.py'}
    if not core.issubset(protocol['runtime_source_sha256']):raise ValueError('Core runtime was not frozen')
    if protocol.get('cost_bps_times_stock_L1')!=10.:raise ValueError('Primary cost protocol differs')
    if protocol.get('signal_lag_closes')!=1:raise ValueError('Final execution requires a full one-close signal delay')
    for prefix in ('method_choice','method_decision'):
        if sha(checked_path(protocol[prefix+'_file']))!=protocol[prefix+'_sha256']:
            raise ValueError('Recorded method decision changed after freeze')
    if sha(checked_path(protocol['publication_metadata_file']))!=protocol['publication_metadata_sha256']:
        raise ValueError('Base-publication chronology evidence changed')
    if sha(checked_path(protocol['data']))!=protocol['dataset_sha256']:raise ValueError('Dataset changed after freeze')
    for name,expected in protocol['runtime_source_sha256'].items():
        if sha(Path(__file__).parent/name)!=expected:raise ValueError('Runtime changed after freeze: '+name)
    for name,expected in protocol['runtime_contract_sha256'].items():
        if sha(Path(__file__).parent/name)!=expected:raise ValueError('Base contract changed after freeze: '+name)
    if sha(ROOT/'ENVIRONMENT_LOCK_20261107.txt')!=protocol['environment_lock_sha256']:
        raise ValueError('Environment record changed after freeze')
    observed=subprocess.run([sys.executable,'-m','pip','freeze'],capture_output=True,text=True,check=True).stdout
    expected=(ROOT/'ENVIRONMENT_LOCK_20261107.txt').read_text(encoding='utf-8-sig')
    if sorted(observed.splitlines())!=sorted(expected.splitlines()):
        raise ValueError('Actual installed Python packages differ from the frozen environment')
    numerical={'float32_matmul_precision':torch.get_float32_matmul_precision(),
        'cuda_matmul_allow_tf32':torch.backends.cuda.matmul.allow_tf32,
        'cudnn_allow_tf32':torch.backends.cudnn.allow_tf32,'cudnn_benchmark':torch.backends.cudnn.benchmark,
        'deterministic_algorithms':torch.are_deterministic_algorithms_enabled()}
    if numerical!=protocol['numerical_backend']:raise ValueError('Numerical backend differs from the frozen protocol')
    for entry in protocol['entries']:
        if 'selection_file' in entry and sha(checked_path(entry['selection_file']))!=entry['selection_sha256']:
            raise ValueError('Frozen selection record changed')
        if entry['kind']=='internvl':
            package=checked_path(entry['package']);manifest=validate_package(package)
            if sha(package/'manifest.json')!=entry['manifest_sha256']:raise ValueError('Frozen model changed')
            if manifest['run'].get('train_end','9999')>='2025-01-01':raise ValueError('Training overlaps test')
        elif sha(checked_path(entry['weights']))!=entry['sha256']:raise ValueError('Frozen control changed')


def assert_no_other_campaign_gpu_jobs():
    names={'train.py','warmup_finance.py','train_covariance_teacher.py','train_covariance_risk.py','train_utility.py',
           'numeric_utility_control.py','numeric_covariance_control.py',
           'integration_smoke.py','benchmark_runtime.py','audit_backend_precision.py','inference.py','evaluate_policy.py',
           'local_gpu_gate.py','evaluate_candidate.py','evaluate_sensitivity.py','evaluate_lora_ablation.py'}
    for process in psutil.process_iter(['pid','name','cmdline']):
        try:
            if process.pid==psutil.Process().pid or 'python' not in (process.info['name'] or '').lower():continue
            arguments=process.info['cmdline'] or []
            if any(Path(argument).name in names for argument in arguments):
                raise RuntimeError('Another campaign compute job is active: '+str(process.pid))
        except (psutil.NoSuchProcess,psutil.AccessDenied):continue


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--protocol',required=True);p.add_argument('--resume',action='store_true')
    args=p.parse_args();protocol=json.loads(Path(args.protocol).read_text())
    configure_finance_precision()
    verify(protocol);assert_no_other_campaign_gpu_jobs();wait_for_compute_window()
    lock=ROOT/'reserved_finance_2025_test.lock.json'
    if (Path(__file__).parent/'reserved_finance_2025_test.lock.json').exists():
        raise RuntimeError('Another final-test lock exists in the old driver directory; inspect it first')
    output=ROOT/'campaign_runs'/('reserved_post_publication_'+protocol['protocol_sha256'][:12])
    if args.resume:
        if not lock.exists():raise RuntimeError('Cannot resume a test that has not been locked')
        prior=json.loads(lock.read_text())
        if prior['protocol_sha256']!=protocol['protocol_sha256']:raise RuntimeError('Cannot resume with different frozen models/protocol')
        if not output.exists():raise RuntimeError('Missing locked output folder')
    else:
        if output.exists():raise FileExistsError('Existing final-block outputs require inspection, not a fresh test')
        with lock.open('x',encoding='utf-8') as stream:
            json.dump({'protocol_sha256':protocol['protocol_sha256'],'started_utc':datetime.now(timezone.utc).isoformat(),
                'output':str(output),'meaning':'Final block now used; no subsequent retuning as if unseen.',
                'history_limit':protocol['history_limit']},stream,indent=2)
        output.mkdir()
        write_json(output/'frozen_protocol.json',protocol)
    status={'status':'running','protocol_sha256':protocol['protocol_sha256'],'completed':[],'resumed':args.resume}
    write_json(output/'status.json',status)
    shared=None;results={};cached={}
    data=str(checked_path(protocol['data']))
    try:
        for entry in protocol['entries']:
            wait_for_compute_window();name=entry['name'];target=output/(name+'.json')
            identity=entry.get('manifest_sha256',entry.get('sha256'))
            if target.exists():
                result=json.loads(target.read_text())
                if result.get('protocol_sha256')!=protocol['protocol_sha256'] or result.get('artifact_sha256')!=identity:
                    raise RuntimeError('Existing test result does not match sealed experiment')
            elif identity in cached:
                result={**cached[identity],'name':name,'alias_of_identical_frozen_model':cached[identity]['name']}
                write_json(target,result)
            else:
                if entry['kind']=='internvl':
                    torch.set_num_threads(2)
                    if shared is None:shared=SharedBackbone.load('cuda','fp32')
                    if shared.adapter_names:
                        base_id=id(shared.base);shared.model=shared.model.unload()
                        assert id(shared.base)==base_id
                        if hasattr(shared.base,'peft_config'):delattr(shared.base,'peft_config')
                    package=checked_path(entry['package'])
                    env_cfg=json.loads((package/'environment_config.json').read_text())
                    policy=InternVLFinancePolicy.load_artifacts(package,shared=shared,device='cuda')
                    reward=FinanceRewardConfig(**env_cfg['reward'])
                    if reward.transaction_cost_bps!=10.:raise ValueError('Primary cost convention changed')
                    device='cuda'
                else:
                    torch.set_num_threads(1)
                    # The ticker count is verified below against the common panel.
                    assets=len(json.loads((checked_path(next(e['package'] for e in protocol['entries'] if e['kind']=='internvl'))/'manifest.json').read_text())['tickers'])
                    policy=NumericCovariancePolicy(assets,attention=entry['attention'])
                    policy.load_state_dict(torch.load(checked_path(entry['weights']),map_location='cpu',weights_only=True),strict=True)
                    policy.eval();reward=FinanceRewardConfig(rebalance_interval=21);device='cpu'
                env=PortfolioEnv(data,start_date=protocol['test_start'],end_date=protocol['test_end'],
                    covariance_window=252,reward_config=reward)
                if entry['kind']=='internvl' and policy.tickers!=env.tickers:raise ValueError('Test ticker order differs')
                allocations=[];infos=lagged_episode(policy,env,device,1,observer=allocation_observer(allocations,1))
                if not all(protocol['test_start']<=r['return_date']<=protocol['test_end'] for r in infos):
                    raise RuntimeError('Unexpected test return boundary')
                result={'name':name,'protocol_sha256':protocol['protocol_sha256'],'artifact_sha256':identity,
                    'metrics':metrics_from_infos(infos),'daily':infos,'role':'frozen reserved-block result',
                    'allocation_quality':allocations,'allocation_quality_summary':{
                        'rebalance_decisions':len(allocations),
                        'mean_variance_to_numerical_optimizer':float(np.mean([r['variance_to_numerical_optimizer'] for r in allocations])),
                        'mean_volatility_to_numerical_optimizer':float(np.mean([r['volatility_to_numerical_optimizer'] for r in allocations])),
                        'scope':'Known decision-close estimated covariance, not an optimum for future realized risk; same cap and fully invested long-only constraint.'}}
                write_json(target,result);del policy;gc.collect()
                if device=='cuda':torch.cuda.empty_cache()
            cached[identity]=result;results[name]=result['metrics'];status['completed'].append(name)
            write_json(output/'status.json',status)
            print(name,json.dumps(result['metrics']),flush=True)
        for name in protocol['classical_controls']:
            wait_for_compute_window();target=output/(name+'.json')
            if target.exists():
                result=json.loads(target.read_text())
                if result.get('protocol_sha256')!=protocol['protocol_sha256']:raise RuntimeError('Existing classical test result differs')
            else:
                control=(static_risk_episode(protocol['test_start'],protocol['test_end'],10.,data,signal_lag=1)
                    if name=='fixed_initial_risk252_monthly' else
                    episode(name,protocol['test_start'],protocol['test_end'],10.,data=data,signal_lag=1))
                result={**control,
                    'protocol_sha256':protocol['protocol_sha256'],'name':name,'role':'frozen classical final-block control'}
                write_json(target,result)
            results[name]=result['metrics'];status['completed'].append(name);write_json(output/'status.json',status)
            print(name,json.dumps(result['metrics']),flush=True)
        write_json(output/'summary.json',{'protocol_sha256':protocol['protocol_sha256'],'results':results,
            'history_limit':protocol['history_limit'],'interpretation':'Report all outcomes, including losses. No post-test model selection.'})
        status.update(status='completed',finished_utc=datetime.now(timezone.utc).isoformat())
    except Exception as error:
        traceback.print_exc();status.update(status='failed',error=str(error));raise
    finally:write_json(output/'status.json',status)
    print('FINAL EVIDENCE:',str(output),flush=True)


if __name__=='__main__':main()
