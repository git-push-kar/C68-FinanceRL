"""Real-package finance router, allocation and cadence parity on development data."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import numpy as np
import torch
from env import PortfolioEnv,FinanceRewardConfig
from evaluate_policy import tensor_obs
from inference import FinanceSpecialist
from model import ModelConfig,write_json,digest
from shared_backbone import SharedBackbone,DomainRouter


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--package',required=True)
    p.add_argument('--data',default='../data/nifty50_prices.csv');p.add_argument('--output',required=True)
    args=p.parse_args();output=Path(args.output);output.mkdir(parents=True,exist_ok=False)
    package=Path(args.package);cfg=ModelConfig(**json.loads((package/'model_config.json').read_text()))
    environment=json.loads((package/'environment_config.json').read_text())
    torch.set_num_threads(2);shared=SharedBackbone.load('cuda',cfg.dtype)
    specialist=FinanceSpecialist(package,shared);router=DomainRouter(shared);specialist.register(router)
    env=PortfolioEnv(args.data,start_date='2023-01-01',end_date='2023-01-10',
        window=cfg.window,covariance_window=cfg.covariance_window,
        reward_config=FinanceRewardConfig(**environment['reward']))
    obs,_=env.reset();start=env.index;env.index=start-1;previous=env._observation();env.index=start
    request={'tickers':env.tickers,'observation':{k:v.tolist() for k,v in previous.items()},'trading_step':0,
        'signal_lag_closes':1,'signal_close_date':str(env.dates[start-1])[:10],
        'execution_close_date':str(env.dates[start])[:10],'execution_portfolio':obs['portfolio'].tolist()}
    write_json(output/'request.json',request);prediction=router.predict('finance',request)
    with torch.inference_mode():scores,_=specialist.policy(tensor_obs(previous,'cuda'))
    expected=env._target_weights(scores[0].cpu().numpy())
    error=float(np.max(np.abs(expected-np.asarray(prediction['weights']))))
    if error>1e-7:raise AssertionError('Router allocation differs from research simulator')
    signal=obs;obs,_,_,_,info=env.step(scores[0].cpu().numpy())
    second=router.predict('finance',{'tickers':env.tickers,'observation':{k:v.tolist() for k,v in signal.items()},'trading_step':1,
        'signal_lag_closes':1,'signal_close_date':info['decision_date'],
        'execution_close_date':str(env.dates[env.index])[:10],'execution_portfolio':obs['portfolio'].tolist()})
    if second['rebalance_due'] or not np.allclose(second['weights'],env.weights,rtol=0,atol=1e-7):
        raise AssertionError('Off-schedule inference does not preserve drifted holdings')
    write_json(output/'prediction.json',prediction)
    write_json(output/'result.json',{'status':'passed','manifest_sha256':digest(package/'manifest.json'),
        'same_shared_instance':router.shared is specialist.policy.shared,
        'allocation_max_difference':error,'off_schedule_holds_actual_weights':True,
        'signal_lag_closes':1,
        'decision_date':info['decision_date'],'test2025_accessed':False,
        'registered_domains':list(router.handlers),'scope':'Real finance package only; other-domain integration not tested'})
    print(json.dumps(json.loads((output/'result.json').read_text()),indent=2))


if __name__=='__main__':main()
