"""Train-only four-logit optimizer and transparent reference for native LoRA."""
import json,os,sys
from pathlib import Path
R=Path(__file__).parent;os.environ['CUDA_VISIBLE_DEVICES']='-1';sys.path.insert(0,str(R/'rl_candidate_static_native'));os.chdir(R/'rl_candidate_static_native')
import numpy as np
import torch
from candidate_market import Market,evaluate
from candidate_path import path_returns
from direct_ledger import negative_sharpe
from execution_policy import wait_for_compute_window
torch.set_num_threads(2)
out=R/'reports/training_allocation_control';out.mkdir(exist_ok=False)
protocol={'scope':'Numerical optimizer, not an adapter; historical development. No reserved data.','training':'2019-2021','validation':'2022','development':'2023-2024','updates':256,'lr':.05,'objective':'negative training Sharpe + .01 KL to same original native distribution','selection':'validation Sharpe at 0 and every16, earliest tie','warmup_target':'minimum penalized training objective among all iterates; never development or validation selected'}
(out/'protocol.json').write_text(json.dumps(protocol,indent=2))
base=json.loads((R/'campaign_runs/nifty_static_native_seed42_64/development_0.json').read_text())['probabilities'][0]
ref=torch.tensor(base,dtype=torch.float64);logits=torch.log((ref-.0125)/.95).requires_grad_();opt=torch.optim.Adam([logits],lr=.05)
train=Market('../data/nifty50_prices.csv','2019-01-01','2021-12-31');val=Market('../data/nifty50_prices.csv','2022-01-01','2022-12-31');dev=Market('../data/nifty50_prices.csv','2023-01-01','2024-12-31')
state=train.reset();e=train.env;begin=e.index;weights=e.weights.copy();cash=e.cash_weight;targets=[]
while True:
 _,t=state;targets.append(t);state,_,done=train.step(np.asarray(base)@t)
 if done:break
rel=e.prices[begin+1:]/e.prices[begin:-1];targets=np.asarray(targets);candidates=[];best=None
for step in range(257):
 wait_for_compute_window();p=.95*logits.softmax(-1)+.0125
 returns=path_returns(p.unsqueeze(0).expand(len(targets),-1),targets,weights,cash,rel)
 objective=negative_sharpe(returns)+.01*(p*(p.log()-ref.log())).sum()
 row={'step':step,'probabilities':p.detach().tolist(),'training_objective':float(objective.detach())}
 if best is None or row['training_objective']<best['training_objective']:best=row.copy()
 if step%16==0:
  pr=p.detach().numpy();v=evaluate(None,val,'cpu',fixed=pr);d=evaluate(None,dev,'cpu',fixed=pr)
  row.update(validation=v['metrics'],development=d['metrics']);candidates.append(row)
  print(json.dumps(row),flush=True)
 if step<256:opt.zero_grad();objective.backward();opt.step()
chosen=max(candidates,key=lambda x:(x['validation']['sharpe_proxy'],-x['step']))
(out/'results.json').write_text(json.dumps({'candidates':candidates,'selected':chosen,'training_only_warmup_target':best},indent=2))
