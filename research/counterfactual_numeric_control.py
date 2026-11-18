"""Three-seed CPU numerical control under the frozen counterfactual-RL protocol.

Same25 state features, plus pooled cross-stock moments, same SFT/RL examples,
updates, execution, selection years and test deferral. This is a control, never
a substitute for the requested InternVL3-2B policy.
"""
import argparse,copy,json,os,time,traceback
from contextlib import contextmanager
from pathlib import Path
import numpy as np
import torch
from torch import nn
from research.counterfactual_rl.market import Market,evaluate,baseline,project_scores,LEVELS
from research.counterfactual_rl.learning import dataset,assess,passes,Sampler,counterfactuals,objective
from research.counterfactual_rl.train import write,validate

class NumericPolicy(nn.Module):
 def __init__(self):
  super().__init__();self.net=nn.Sequential(nn.Linear(75,128),nn.Tanh(),nn.Linear(128,64),nn.Tanh(),nn.Linear(64,7));self.reference=None
 def encode(self,obs):
  rows=sorted(obs['stocks'],key=lambda r:r['ticker']);n=len(rows);assert n==49;eq=obs['equity'];ranks={k:np.argsort(np.argsort([r[k] for r in rows],kind='stable'),kind='stable')/(n-1) for k in ['return20','return60','return120','vol20']};x=[]
  for i,r in enumerate(rows):
   x.append([np.log1p(r['historical_quote'])/12,np.log1p(r['held_shares'])/12,r['position_value']/eq,r['weight'],r['affordable_shares_cash']*r['historical_quote']/eq,r['return5']/.1,r['return20']/.2,r['return60']/.4,r['return120']/.6,r['vol20']/.4,r['drawdown60']/.4,np.log(max(r['volume_ratio'],.01)),np.log1p(obs['initial_capital'])/12,np.log1p(obs['cash'])/12,np.log1p(eq)/12,obs['cash']/eq,obs['max_stock_weight'],obs['fees_bps']/100,obs['slippage_bps']/100,float(r['held_shares']>0)]+[ranks[k][i] for k in ranks]+[r['historical_quote']/eq])
  x=np.clip(np.array(x,np.float32),-10,10);return torch.tensor(np.concatenate([x,np.broadcast_to(x.mean(0),x.shape),np.broadcast_to(x.std(0),x.shape)],axis=1))
 def forward(self,x):return .95*self.net(x).softmax(1)+.05/7
 def choose(self,obs):
  with torch.no_grad():p=self(self.encode(obs)).numpy()
  return p.argmax(1),{'mode':'numerical_control','probabilities':p.tolist()}
 def freeze_reference(self):self.reference=copy.deepcopy(self.state_dict())
 @contextmanager
 def reference_context(self):
  current=copy.deepcopy(self.state_dict());self.load_state_dict(self.reference)
  try:yield
  finally:self.load_state_dict(current)

def run(root):
 protocol=json.loads((root/'protocol.json').read_text());kind=protocol['teacher']['selected'];sft_steps=protocol['args']['sft_steps'];rl_steps=protocol['args']['rl_steps'];m=Market();control=root/'numeric_control';control.mkdir(exist_ok=False)
 write(control/'protocol.json',{'source':'research/counterfactual_numeric_control.py','features':'Same25 features plus permutation-invariant cross-sectional mean and std; no ticker identifier. Shared75->128->64->7 MLP. Native model has ticker context and all individual records; this control pools other-stock context.','parameters':sum(p.numel() for p in NumericPolicy().parameters()),'seeds':[42,123,456],'selection':'Same2023/2024 validation and held-out2022 learning gate; no test-based selection. Equal optimizer-update count, model-specific learning rate1e-3SFT/1e-4RL.'})
 for seed in [42,123,456]:
  out=control/f'seed{seed}';out.mkdir();torch.manual_seed(seed);p=NumericPolicy();torch.save(p.state_dict(),out/'untrained.pt');candidates=[]
  def checkpoint(name,stage):
   torch.save(p.state_dict(),out/f'{name}.pt');score,r=validate(p,m);write(out/f'{name}_validation.json',r);entry={'checkpoint':name,'stage':stage,'score':score,'years':{y:v['metrics'] for y,v in r.items()}};candidates.append(entry);write(out/'candidates.json',candidates)
  checkpoint('untrained','untrained');rows=dataset(m,sft_steps,seed,kind);gate=dataset(m,64,seed+77,kind,gate=True);encoded=[p.encode(r['observation']) for r in rows];ge=[p.encode(r['observation']) for r in gate];raw=assess(p,gate,ge);write(out/'raw_gate.json',raw);counts=np.bincount(np.array([r['actions'] for r in rows]).ravel(),minlength=7);cw=torch.tensor(1/np.maximum(counts,1)**.5,dtype=torch.float32);cw/=cw.mean();opt=torch.optim.AdamW(p.parameters(),lr=.001,weight_decay=0.);history=[]
  for step,j in enumerate(np.random.default_rng(seed+100).permutation(len(rows)),1):
   prob=p(encoded[j]);labels=torch.tensor(rows[j]['actions']);target=.97*nn.functional.one_hot(labels,7)+.03/7;weight=cw[labels];loss=(-(target*prob.log()).sum(1)*weight).sum()/weight.sum();opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(p.parameters(),1.);opt.step()
   if step%256==0 or step==len(rows):
    score=assess(p,gate,ge);history.append({'step':step,'passed':passes(score,raw),**score});write(out/'gate_history.json',history);torch.save(p.state_dict(),out/f'sft_{step}.pt')
  eligible=[g for g in history if g['passed']]
  if not eligible:write(out/'training_done.json',{'state':'failed_learning_gate'});continue
  chosen=min(eligible,key=lambda g:(g['allocation_l1'],g['step']));p.load_state_dict(torch.load(out/f"sft_{chosen['step']}.pt",weights_only=True));checkpoint('supervised','supervised');p.freeze_reference();sampler=Sampler(m,seed+10);opt=torch.optim.AdamW(p.parameters(),lr=.0001,weight_decay=0.);account=None
  for step in range(1,rl_steps+1):
   if account is None:t,capital=sampler.start();cash=capital;q=np.zeros(49,int);pending=np.zeros(2);decision=0
   else:t,capital,cash,q,pending,decision=account
   obs=m.observe(t,cash,q,capital,pending);e=p.encode(obs)
   with torch.no_grad():
    old=p(e).clone();actions=torch.distributions.Categorical(old).sample().numpy()
    with p.reference_context():ref=p(e).clone()
   rewards,result=counterfactuals(m,t,cash,q,pending,actions,t+20);prob=p(e);assert float(abs(prob.detach()-old).max())<2e-6;loss,stats=objective(prob,old,ref,rewards);opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(p.parameters(),1.);opt.step();decision+=1;account=(t+20,capital,result['cash'],result['qty'],result['pending'],decision) if decision<3 else None
   if step%32==0:
    write(control/'status.json',{'state':'counterfactual_rl','seed':seed,'step':step,'total':rl_steps,'pid':os.getpid(),**stats});torch.save({'model':p.state_dict(),'optimizer':opt.state_dict(),'reference':p.reference,'step':step,'account':account,'rng':sampler.rng.bit_generator.state,'torch_rng':torch.get_rng_state()},out/'recovery.pt')
   if step%128==0 or step==rl_steps:checkpoint(f'rl_{step}','rl')
  selected=max(candidates,key=lambda x:x['score']);write(out/'selection.json',selected);write(out/'training_done.json',{'state':'trained','selected':selected})
 # Never inspect the final financial test before all native seeds finish training.
 while not all((root/f'seed{s}'/'training_done.json').exists() for s in [42,123,456]):
  write(control/'status.json',{'state':'waiting_for_native_training','pid':os.getpid()})
  if json.loads((root/'status.json').read_text()).get('state')=='failed':raise RuntimeError('Native campaign failed; test evaluation remains deferred')
  time.sleep(30)
 for seed in [42,123,456]:
  out=control/f'seed{seed}';done=json.loads((out/'training_done.json').read_text())
  if done['state']!='trained':continue
  p=NumericPolicy();metrics=[]
  for name,file in [('selected',done['selected']['checkpoint']),('supervised','supervised'),('untrained','untrained')]:
   p.load_state_dict(torch.load(out/f'{file}.pt',weights_only=True))
   for capital in [500,5000,50000,100000,200000]:
    r=evaluate(m,p.choose,'2025-01-01','2026-01-30',capital);write(out/f'test_{capital}_{name}.json',r);metrics.append({'policy':name,'capital':capital,**r['metrics']})
  write(out/'test_metrics.json',metrics)
 write(control/'status.json',{'state':'completed','pid':os.getpid()})

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);args=ap.parse_args();torch.set_num_threads(2);root=Path(args.run)
 try:run(root)
 except BaseException as exc:
  (root/'numeric_control').mkdir(exist_ok=True);write(root/'numeric_control/status.json',{'state':'failed','error':repr(exc),'traceback':traceback.format_exc()});raise
