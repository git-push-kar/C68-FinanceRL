"""Compare inherited cuDNN TF32 permission with both TF32 flags disabled.

No optimizer step or model selection. The training stack's original defaults
are reported rather than conflating FP32 tensor dtype with every backend flag.
"""
import argparse
import json
from pathlib import Path
import math
import numpy as np
import torch
from env import PortfolioEnv,FinanceRewardConfig
from model import InternVLFinancePolicy,write_json,digest
from train import lora_digest
from train_covariance_teacher import training_examples,evaluate
from train_covariance_risk import relative_risk_loss
from warmup_finance import label_indices


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--package',required=True)
    p.add_argument('--output',required=True);p.add_argument('--data',default='../data/nifty50_prices.csv')
    args=p.parse_args();output=Path(args.output);output.mkdir(parents=True,exist_ok=False)
    thresholds={'maximum_score_difference':1e-5,'relative_gradient_L2_difference':1e-5,
                'maximum_daily_net_return_difference':1e-8,'absolute_Sharpe_difference':1e-5}
    write_json(output/'plan.json',{'thresholds':thresholds,'no_optimizer_updates':True,'test2025_accessed':False})
    torch.set_num_threads(2);policy=InternVLFinancePolicy.load_artifacts(args.package,device='cuda',for_training=True)
    policy.critic.requires_grad_(False);policy.log_std.requires_grad_(False)
    manifest=json.loads((Path(args.package)/'manifest.json').read_text());before=lora_digest(policy)
    config=json.loads((Path(args.package)/'environment_config.json').read_text());reward=FinanceRewardConfig(**config['reward'])
    env=PortfolioEnv(args.data,end_date='2022-12-31',covariance_window=252,reward_config=reward);env.reset()
    _,holdout,_=label_indices(len(env.prices),252,0);probe=training_examples(env,holdout[np.linspace(0,len(holdout)-1,32,dtype=int)])
    obs,targets=probe;rows={};saved=(torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32)
    try:
        for enabled in (True,False):
            torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=enabled
            policy.zero_grad(set_to_none=True)
            one={k:v[:1].to('cuda') for k,v in obs.items()};scores,_=policy(one)
            if manifest['run']['algorithm'].startswith('causal_log_relative'):
                loss=relative_risk_loss(scores,one['covariance'])
            else:loss=-(targets[:1].to('cuda')*torch.log_softmax(scores,-1)).sum()
            loss.backward()
            gradients={n:p.grad.detach().cpu().clone() for n,p in policy.named_parameters() if p.requires_grad and p.grad is not None}
            with torch.no_grad():
                predictions=torch.cat([policy({k:v[i:i+1].to('cuda') for k,v in obs.items()})[0].cpu() for i in range(len(targets))])
            development=PortfolioEnv(args.data,start_date='2023-01-01',end_date='2024-12-31',covariance_window=252,reward_config=reward)
            metrics=evaluate(policy,development,'cuda',output/f'development_tf32_{enabled}.json')
            rows[enabled]={'gradients':gradients,'scores':predictions,'loss':float(loss.detach()),'metrics':metrics}
        if rows[True]['gradients'].keys()!=rows[False]['gradients'].keys():raise AssertionError('Trainable gradient sets differ')
        difference=sum(float((rows[True]['gradients'][k].double()-rows[False]['gradients'][k].double()).square().sum()) for k in rows[True]['gradients'])
        reference=sum(float(v.double().square().sum()) for v in rows[True]['gradients'].values())
        relative=math.sqrt(difference/max(reference,1e-30));score_error=float((rows[True]['scores']-rows[False]['scores']).abs().max())
        daily={flag:json.loads((output/f'development_tf32_{flag}.json').read_text())['daily'] for flag in (True,False)}
        daily_error=max(abs(a['net_return']-b['net_return']) for a,b in zip(daily[True],daily[False]))
        sharpe_error=abs(rows[True]['metrics']['sharpe_proxy']-rows[False]['metrics']['sharpe_proxy'])
        unchanged=before==lora_digest(policy)
        passed=unchanged and score_error<=thresholds['maximum_score_difference'] and relative<=thresholds['relative_gradient_L2_difference'] and daily_error<=thresholds['maximum_daily_net_return_difference'] and sharpe_error<=thresholds['absolute_Sharpe_difference']
        result={'status':'passed' if passed else 'failed','manifest_sha256':digest(Path(args.package)/'manifest.json'),
            'maximum_score_difference':score_error,'relative_gradient_L2_difference':relative,
            'maximum_daily_net_return_difference':daily_error,'absolute_Sharpe_difference':sharpe_error,
            'LoRA_unchanged':unchanged,'thresholds':thresholds,'test2025_accessed':False,
            'metrics':{str(k):v['metrics'] for k,v in rows.items()},
            'scope':'32 pre-2023 observations, one real full-backbone backward and complete development simulations. This is a sensitivity audit, not proof of bitwise cross-hardware training equality.'}
        write_json(output/'result.json',result);print(json.dumps({k:v for k,v in result.items() if k!='metrics'},indent=2))
    finally:torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32=saved


if __name__=='__main__':main()
