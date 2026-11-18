"""Measure inference cost, including cheaper controls; no performance selection."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from sklearn.covariance import LedoitWolf
from threadpoolctl import threadpool_limits
from classical_panel import min_variance
from env import PortfolioEnv,FinanceRewardConfig
from evaluate_policy import tensor_obs
from execution_policy import wait_for_compute_window
from inference import FinanceSpecialist
from model import ModelConfig,digest,write_json
from numeric_covariance_control import NumericCovariancePolicy
from shared_backbone import SharedBackbone,DomainRouter


def measure(function,repeats,cuda=False):
    samples=[]
    for _ in range(5):function()
    for _ in range(repeats):
        wait_for_compute_window()
        if cuda:torch.cuda.synchronize()
        start=time.perf_counter();function()
        if cuda:torch.cuda.synchronize()
        samples.append(1000*(time.perf_counter()-start))
    return {'median_ms':float(np.median(samples)),'p95_ms':float(np.quantile(samples,.95)),
        'mean_ms':float(np.mean(samples)),'samples':repeats}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--package',required=True)
    p.add_argument('--numeric-weights',required=True);p.add_argument('--output',required=True)
    p.add_argument('--data',default='../data/nifty50_prices.csv');p.add_argument('--repeats',type=int,default=50)
    args=p.parse_args();output=Path(args.output)
    if output.exists():raise FileExistsError(output)
    torch.set_num_threads(2);package=Path(args.package)
    cfg=ModelConfig(**json.loads((package/'model_config.json').read_text()))
    environment=json.loads((package/'environment_config.json').read_text())
    wait_for_compute_window();start=time.perf_counter();shared=SharedBackbone.load('cuda',cfg.dtype)
    specialist=FinanceSpecialist(package,shared);load_seconds=time.perf_counter()-start
    router=DomainRouter(shared);specialist.register(router)
    env=PortfolioEnv(args.data,start_date='2023-01-01',end_date='2023-01-10',
        covariance_window=252,reward_config=FinanceRewardConfig(**environment['reward']))
    current,_=env.reset();start_index=env.index;env.index-=1;obs=env._observation();env.index=start_index
    gpu_obs=tensor_obs(obs,'cuda');cpu_obs=tensor_obs(obs,'cpu')
    request={'tickers':env.tickers,'observation':{k:v.tolist() for k,v in obs.items()},'trading_step':0,
        'signal_lag_closes':1,'signal_close_date':str(env.dates[start_index-1])[:10],
        'execution_close_date':str(env.dates[start_index])[:10],'execution_portfolio':current['portfolio'].tolist()}
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        forward=measure(lambda:specialist.policy(gpu_obs),args.repeats,cuda=True)
        api=measure(lambda:router.predict('finance',request),args.repeats,cuda=True)
    torch.set_num_threads(1)
    numeric=NumericCovariancePolicy(env.n_assets,attention=True)
    numeric.load_state_dict(torch.load(args.numeric_weights,map_location='cpu',weights_only=True),strict=True);numeric.eval()
    cov=obs['covariance'].astype(np.float64);history=env.prices[start_index-1-252:start_index]
    returns=history[1:]/history[:-1]-1
    with threadpool_limits(limits=1),torch.inference_mode():
        small=measure(lambda:numeric(cpu_obs),args.repeats)
        optimizer=measure(lambda:min_variance(cov),args.repeats)
        covariance=measure(lambda:LedoitWolf().fit(returns).covariance_,args.repeats)
    result={'hardware':torch.cuda.get_device_name(),'dtype_active':cfg.dtype,
        'manifest_sha256':digest(package/'manifest.json'),'numeric_weights_sha256':digest(args.numeric_weights),
        'observed_load_seconds':load_seconds,'gpu_policy_forward':forward,'gpu_router_request':api,
        'cpu_small_transformer_forward':small,'cpu_minimum_variance_solve':optimizer,
        'cpu_covariance_estimation':covariance,'gpu_peak_allocated_GiB':torch.cuda.max_memory_allocated()/2**30,
        'gpu_reserved_GiB':torch.cuda.memory_reserved()/2**30,'test2025_accessed':False,
        'scope':'One fixed first-development-close input; GPU timings synchronize CUDA, policy inputs are resident; router includes validation/transfer/allocation. CPU controls use one math thread. CSV/features/covariance construction excluded from policy timings and covariance is timed separately. Load may benefit from warm filesystem cache. Excludes desktop/driver VRAM overhead.',
        'interpretation':'Report slower and larger inference when present. Shared-backbone integration does not imply a standalone financial-compute advantage.'}
    output.parent.mkdir(parents=True,exist_ok=True);write_json(output,result);print(json.dumps(result,indent=2))


if __name__=='__main__':main()
