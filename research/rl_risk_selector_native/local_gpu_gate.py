"""Measured full-pretrained gate; never touches held-out 2025 returns."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import time

import torch

from config import ModelConfig, PPOConfig, serializable
from env import FinanceRewardConfig, PortfolioEnv
from evaluate_policy import tensor_obs
from model import InternVLFinancePolicy, write_json
from train import lora_digest, seed_everything


def frozen_digest(shared):
    checksum = hashlib.sha256()
    for name, parameter in sorted(shared.named_parameters()):
        if 'lora_' not in name:
            checksum.update(name.encode())
            checksum.update(memoryview(parameter.detach().contiguous().view(torch.uint8).cpu().numpy()))
    return checksum.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True)
    p.add_argument('--data',default=str(Path(__file__).resolve().parent.parent/'data/nifty50_prices.csv'))
    args=p.parse_args()
    output=Path(args.output)
    output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2)
    seed_everything(42)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    started=time.monotonic()
    env=PortfolioEnv(args.data,end_date='2022-12-31')
    policy=InternVLFinancePolicy(ModelConfig(dtype='fp32',actor_init_scale=.01,score_scale=1.,
        use_asset_ids=False,normalize_market=True),env.tickers,device='cuda')
    torch.cuda.reset_peak_memory_stats()
    frozen_before=frozen_digest(policy.shared)
    lora_before=lora_digest(policy)
    obs,_=env.reset()
    obs=tensor_obs(obs,'cuda')
    with torch.no_grad():
        action,old_logp,_=policy.act(obs)
    logp,values,_=policy.evaluate_actions(obs,action)
    difference=float((logp-old_logp).abs().max().detach())
    if difference > .01: raise RuntimeError('Likelihood parity failed')
    optimizer=torch.optim.Adam([p for p in policy.parameters() if p.requires_grad],lr=1e-5)
    optimizer.zero_grad(set_to_none=True)
    # A nonconstant numerical target proves actual LoRA/encoder optimization.
    scores,_=policy(obs)
    target=torch.linspace(-.5,.5,len(env.tickers),device='cuda').view(1,-1)
    loss=(scores-target).square().mean()+values.square().mean()
    loss.backward()
    if any(p.grad is not None for n,p in policy.shared.named_parameters() if 'lora_' not in n):
        raise RuntimeError('Frozen pretrained base received gradients')
    lora_gradient=sum(float(p.grad.detach().abs().sum()) for n,p in policy.shared.named_parameters()
                      if 'lora_' in n and p.grad is not None)
    if lora_gradient<=0: raise RuntimeError('Missing LoRA gradients')
    torch.nn.utils.clip_grad_norm_([p for p in policy.parameters() if p.requires_grad],1.)
    optimizer.step()
    lora_after=lora_digest(policy)
    frozen_after=frozen_digest(policy.shared)
    assert frozen_before==frozen_after
    assert lora_before!=lora_after
    with torch.no_grad(): expected=policy(obs)[0].cpu()
    package=output/'package'
    policy.save_artifacts(package,serializable(PPOConfig()),serializable(FinanceRewardConfig()),
        {'algorithm':'hardware_gate_not_financial_training','train_end':'2022-12-31',
         'optimizer_updates':1,'learning_verified':True,'initial_lora_sha256':lora_before,'final_lora_sha256':lora_after})
    shared=policy.shared
    del loss,scores,values,logp,optimizer,policy
    gc.collect(); torch.cuda.empty_cache()
    # Unwrap without merging. Leaving a PEFT object with zero adapters makes
    # its active_peft_config invalid in PEFT 0.15; retain the same base object.
    base_identity=id(shared.base)
    shared.model=shared.model.unload()
    assert id(shared.base)==base_identity
    restored=InternVLFinancePolicy.load_artifacts(package,shared=shared,device='cuda')
    with torch.no_grad(): actual=restored(obs)[0].cpu()
    torch.testing.assert_close(actual,expected,rtol=1e-5,atol=1e-6)
    result={'status':'passed','backbone':'full pinned InternVL3-2B','dtype_active_path':'fp32',
        'execution_mode':shared.execution_mode,'lora_gradient_l1':lora_gradient,
        'frozen_before':frozen_before,'frozen_after':frozen_after,
        'lora_changed':lora_before!=lora_after,'max_initial_likelihood_difference':difference,
        'reload_max_score_difference':float((actual-expected).abs().max()),
        'peak_allocated_GiB':torch.cuda.max_memory_allocated()/2**30,
        'peak_reserved_GiB':torch.cuda.max_memory_reserved()/2**30,
        'seconds':time.monotonic()-started,'financial_benefit_tested':False,
        'source_weight_identity':shared.source_identity}
    write_json(output/'result.json',result)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__': main()
