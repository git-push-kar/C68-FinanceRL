"""Constant-context original-token LoRA, trained by full-path financial reward."""
import argparse,time,json
from pathlib import Path
import numpy as np
import torch
from candidate_market import Market,evaluate,tensor
from candidate_path import path_returns
from direct_ledger import negative_sharpe
from text_policy import TextPolicy
from train import seed_everything,lora_digest
from model import write_json,digest
from execution_policy import configure_finance_precision,wait_for_compute_window

def main():
 a=argparse.ArgumentParser();a.add_argument('--output',required=True);a.add_argument('--seed',type=int,default=42);a.add_argument('--updates',type=int,default=64);a.add_argument('--eval-every',type=int,default=4);args=a.parse_args()
 wait_for_compute_window();seed_everything(args.seed);configure_finance_precision();torch.set_num_threads(2)
 out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
 data='../multiasset_data_20261115/five_asset_development_through2024.csv';train=Market(data,'2010-01-01','2018-12-31');val=Market(data,'2019-01-01','2021-12-31');dev=Market(data,'2022-01-01','2024-12-31')
 policy=TextPolicy(train.env.tickers);obs=tensor(train.reset()[0],'cuda')
 params=[p for p in policy.parameters() if p.requires_grad]
 if any('lora_' not in n for n,p in policy.named_parameters() if p.requires_grad):raise AssertionError('Non-LoRA parameter')
 optimizer=torch.optim.AdamW([{'params':[p for n,p in policy.named_parameters() if p.requires_grad and '.lm_head.' not in n],'lr':1e-5},{'params':[p for n,p in policy.named_parameters() if p.requires_grad and '.lm_head.' in n],'lr':3e-4}],weight_decay=0)
 def probabilities():
  p=policy(obs)[0].probs[0].double();return p/p.sum()
 with torch.no_grad():reference=probabilities().detach()
 initial=lora_digest(policy);candidates=[]
 metadata={'algorithm':'constant_context_direct_financial_reinforcement','arguments':vars(args),'training':'2010-2018','selection':'2019-2021 net Sharpe, zero eligible, earliest tie','development':'2022-2024 reused','execution':'expected strategy mixture, exact 10bp fees; no state-dependent timing','objective':'negative full-training net Sharpe + .01 KL to original policy','dataset_sha256':digest(data),'source_sha256':{p.name:digest(p) for p in Path(__file__).parent.glob('*.py')},'test2026_accessed':False}
 write_json(out/'protocol.json',metadata)
 def assess(step):
  with torch.no_grad():p=probabilities().cpu().numpy()
  v=evaluate(None,val,'cpu',fixed=p);d=evaluate(None,dev,'cpu',fixed=p)
  write_json(out/f'validation_{step}.json',v);write_json(out/f'development_{step}.json',d)
  package=out/f'checkpoint_{step}';policy.save(package,metadata)
  candidates.append({'step':step,'validation':v['metrics'],'development':d['metrics'],'package':str(package.resolve())});write_json(out/'candidates.json',candidates)
 assess(0);started=time.monotonic()
 for step in range(1,args.updates+1):
  wait_for_compute_window()
  with torch.no_grad():prior=probabilities().cpu().numpy()
  state=train.reset();e=train.env;begin=e.index;weights=e.weights.copy();cash=e.cash_weight;targets=[]
  while True:
   _,t=state;targets.append(t);state,_,done=train.step(prior@t)
   if done:break
  leaves=torch.tensor(prior,dtype=torch.float64,requires_grad=True)
  rel=e.prices[begin+1:]/e.prices[begin:-1]
  returns=path_returns(leaves.unsqueeze(0).expand(len(targets),-1),np.asarray(targets),weights,cash,rel)
  error=float(np.max(abs(returns.detach().numpy()-np.asarray([i['net_return'] for i in train.infos]))))
  if error>1e-11:raise AssertionError('Ledger reconstruction mismatch')
  objective=negative_sharpe(returns);coefficient=torch.autograd.grad(objective,leaves)[0]
  optimizer.zero_grad();p=probabilities()
  discrepancy=float(np.max(abs(p.detach().cpu().numpy()-prior)))
  if discrepancy>1e-7:raise AssertionError('Likelihood mismatch')
  kl=(p*(p.log()-reference.log())).sum();loss=(p*coefficient.to(p.device)).sum()+.01*kl
  loss.backward();grad=float(torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True));optimizer.step()
  with torch.no_grad():post=probabilities().cpu().numpy()
  postkl=float(np.sum(prior*np.log(prior/post)))
  if not np.isfinite(postkl) or postkl>.03:raise AssertionError('KL guard')
  row={'update':step,'training_Sharpe_before_update':-float(objective.detach()),'financial_path_max_error':error,'initial_probability_error':discrepancy,'post_KL':postkl,'gradient_norm':grad,'probabilities':post.tolist(),'seconds':time.monotonic()-started}
  write_json(out/'progress.json',row);print(json.dumps(row),flush=True)
  if step%args.eval_every==0 or step==args.updates:assess(step)
 selected=max(candidates,key=lambda c:(c['validation']['sharpe_proxy'],-c['step']))
 write_json(out/'selection.json',{'selected':selected,'initial':candidates[0],'criterion':metadata['selection']})
 write_json(out/'status.json',{'status':'completed','optimizer_updates':args.updates,'initial_lora_sha256':initial,'final_lora_sha256':lora_digest(policy),'financial_goal_met':False})
if __name__=='__main__':main()
