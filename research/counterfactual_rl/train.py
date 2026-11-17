"""Complete preregistered three-seed counterfactual financial-RL campaign."""
import argparse,copy,datetime,hashlib,json,os,random,time,traceback,zipfile,gc
from pathlib import Path
import numpy as np
import torch
from peft import set_peft_model_state_dict
from safetensors.torch import load_file
from .market import Market,evaluate,baseline,project_scores,LEVELS
from .policy import NativePolicy
from .learning import select_teacher,dataset,assess,passes,Sampler,counterfactuals,objective,teacher

ROOT=Path(__file__).resolve().parent
BASE='C:/Users/Parth/Documents/PolyvalentFinance/local_models/InternVL3-2B'

def write(path,data):
 path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_bytes(json.dumps(data,indent=2).encode());tmp.replace(path)

def load(policy,path):
 path=Path(path);manifest=json.loads((path/'manifest.json').read_text())
 for name,digest in manifest['sha256'].items():
  if hashlib.sha256((path/name).read_bytes()).hexdigest()!=digest:raise ValueError('Checkpoint corrupt')
 result=set_peft_model_state_dict(policy.model,load_file(str(path/'finance_lora/finance/adapter_model.safetensors')),adapter_name='finance')
 if result.unexpected_keys:raise ValueError(result.unexpected_keys)
 policy.numeric_bridge.load_state_dict(load_file(str(path/'numeric_bridge.safetensors')))

def optimizer(p,stage):
 rates=(1e-5,5e-5,8e-4) if stage=='sft' else (1e-6,5e-6,2e-5);groups=[[],[],[]]
 for n,v in p.named_parameters():
  if v.requires_grad:groups[2 if n.startswith('numeric_bridge') else 1 if '.lm_head.' in n else 0].append(v)
 return torch.optim.AdamW([{'params':g,'lr':rate} for g,rate in zip(groups,rates)],weight_decay=0.)

def save_recovery(out,p,opt,state,rng):
 data={'parameters':{n:v.detach().cpu() for n,v in p.named_parameters() if v.requires_grad},'optimizer':opt.state_dict(),'state':state,'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),'rng':rng.bit_generator.state,'reference':[v.cpu() for v in getattr(p,'reference_weights',[])]}
 tmp=out/'recovery.tmp';torch.save(data,tmp);tmp.replace(out/'recovery.pt')

def validate(p,m):
 results={str(y):evaluate(m,p.choose,f'{y}-01-01',f'{y}-12-31',100000) for y in [2023,2024]}
 # Two-year mean net log growth; do not choose on the previously seen test.
 score=float(np.mean([np.log1p(v['metrics']['return']) for v in results.values()]))
 return score,results

def probe(p,m):
 t,_=m.bounds('2023-01-01','2023-12-31');obs=m.observe(t,100000,np.zeros(49,int),100000);a,receipt=p.choose(obs);result={}
 for name in ['reorder','capital5000','capital200000','holdings','reverse_risk','reverse_returns']:
  o=copy.deepcopy(obs)
  if name=='reorder':o['stocks'].reverse();o['universe'].reverse()
  elif name.startswith('capital'):
   capital=float(name[7:]);o=m.observe(t,capital,np.zeros(49,int),capital)
  elif name=='holdings':o=m.observe(t,10000,np.ones(49,int),100000)
  elif name=='reverse_risk':
   for r,v in zip(o['stocks'],[r['vol20'] for r in o['stocks']][::-1]):r['vol20']=v
  elif name=='reverse_returns':
   for r in o['stocks']:
    for k in ['return5','return20','return60','return120']:r[k]*=-1
  b,r=p.choose(o);result[name]={'changed_actions':int((a!=b).sum()),'probability_change':float(abs(np.array(receipt['probabilities'])-np.array(r['probabilities'])).max())}
 assert result['reorder']['changed_actions']==0 and result['reorder']['probability_change']==0
 return result

def train_seed(args,root,m,kind,seed,protocol):
 out=root/f'seed{seed}';out.mkdir(exist_ok=True)
 if (out/'training_done.json').exists():return json.loads((out/'training_done.json').read_text())
 torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);np.random.seed(seed);random.seed(seed)
 p=NativePolicy(BASE);params=[v for v in p.parameters() if v.requires_grad];sampler=Sampler(m,seed+10);rng=sampler.rng
 rows=dataset(m,args.sft_steps,seed,kind);gate=dataset(m,64,seed+77,kind,gate=True);encoded=[p.encode(r['observation']) for r in rows];ge=[p.encode(r['observation']) for r in gate]
 write(out/'curriculum.json',rows);write(out/'gate_examples.json',gate)
 with torch.no_grad():pristine=p(encoded[0]).clone()
 candidates=[];gate_history=[];state={'phase':'sft','step':0,'rl_step':0,'account':None};opt=optimizer(p,'sft')
 def checkpoint(name,stage):
  if not (out/name).exists():p.save(out/name,protocol)
  score,result=validate(p,m);write(out/f'{name}_validation.json',result);row={'checkpoint':name,'stage':stage,'score':score,'years':{y:r['metrics'] for y,r in result.items()}};candidates.append(row);write(out/'candidates.json',candidates);print(json.dumps({'phase':'validation','seed':seed,**row}),flush=True)
  return row
 if (out/'recovery.pt').exists():
  saved=torch.load(out/'recovery.pt',map_location='cpu',weights_only=False);state=saved['state'];opt=optimizer(p,state['phase'])
  with torch.no_grad():
   for n,v in p.named_parameters():
    if v.requires_grad:v.copy_(saved['parameters'][n].to(v.device))
  opt.load_state_dict(saved['optimizer']);rng.bit_generator.state=saved['rng'];torch.set_rng_state(saved['torch_rng']);torch.cuda.set_rng_state_all(saved['cuda_rng'])
  if saved['reference']:p.reference_weights=[v.cuda() for v in saved['reference']]
  candidates=json.loads((out/'candidates.json').read_text());gate_history=json.loads((out/'gate_history.json').read_text()) if (out/'gate_history.json').exists() else [];raw=json.loads((out/'raw_gate.json').read_text())
 else:
  raw=assess(p,gate,ge);write(out/'raw_gate.json',raw);checkpoint('untrained','untrained');save_recovery(out,p,opt,state,rng)
 counts=np.bincount(np.array([r['actions'] for r in rows]).ravel(),minlength=7);weights=1/np.maximum(counts,1)**.5;weights/=weights.mean();cw=torch.tensor(weights,dtype=torch.float32,device='cuda');order=np.random.default_rng(seed+100).permutation(len(rows));losses=[]
 if state['phase']=='sft':
  for step in range(state['step'],len(rows)):
   j=int(order[step]);prob=p(encoded[j]);labels=torch.tensor(rows[j]['actions'],device='cuda');target=.97*torch.nn.functional.one_hot(labels,7)+.03/7;weight=cw[labels];loss=(-(target*prob.log()).sum(1)*weight).sum()/weight.sum();opt.zero_grad(set_to_none=True);loss.backward();norm=float(torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True));opt.step();opt.zero_grad(set_to_none=True);losses.append(float(loss.detach()));state['step']=step+1
   if step==0:
    with torch.no_grad():updated=p(encoded[0]).clone()
    delta=float(abs(updated-pristine).max());assert delta>0;p.save(out/'reload_probe',{'purpose':'mechanical correctness'})
    with torch.no_grad():
     for v in params:v.zero_()
    load(p,out/'reload_probe')
    with torch.no_grad():
     err=float(abs(p(encoded[0])-updated).max())
     with p.model.disable_adapter():rawerr=float(abs(p(encoded[0])-pristine).max())
    assert max(err,rawerr)<2e-6;write(out/'reload_receipt.json',{'probability_change':delta,'reload_error':err,'disabled_error':rawerr,'peak_gpu_bytes':torch.cuda.max_memory_allocated()})
   if (step+1)%32==0 or step==0:
    save_recovery(out,p,opt,state,rng);status={'state':'supervised_training','seed':seed,'pid':os.getpid(),'step':step+1,'total':len(rows),'loss':float(np.mean(losses[-32:])),'gradient_norm':norm};write(root/'status.json',status);print(json.dumps(status),flush=True)
   if (step+1)%256==0 or step+1==len(rows):
    score=assess(p,gate,ge);entry={'step':step+1,'passed':passes(score,raw),**score};gate_history.append(entry);write(out/'gate_history.json',gate_history)
    if not (out/f'sft_{step+1}').exists():p.save(out/f'sft_{step+1}',protocol)
    print(json.dumps({'phase':'learning_gate','seed':seed,**entry}),flush=True)
  eligible=[g for g in gate_history if g['passed']]
  if not eligible:
   done={'seed':seed,'state':'failed_learning_gate','history':gate_history};write(out/'training_done.json',done);return done
  gate_selected=min(eligible,key=lambda g:(g['allocation_l1'],g['step']));load(p,out/f"sft_{gate_selected['step']}");write(out/'sft_selection.json',gate_selected);checkpoint('supervised','supervised');p.freeze_reference();write(out/'sft_sensitivity.json',probe(p,m));state['phase']='rl';opt=optimizer(p,'rl');save_recovery(out,p,opt,state,rng)
 for step in range(state['rl_step'],args.rl_steps):
  begin=time.perf_counter()
  if state['account'] is None:
   t,capital=sampler.start();account={'t':t,'capital':capital,'cash':capital,'qty':[0]*49,'pending':[0.,0.],'decision':0}
  else:account=state['account']
  t=account['t'];capital=account['capital'];cash=account['cash'];q=np.array(account['qty'],int);pending=np.array(account['pending']);obs=m.observe(t,cash,q,capital,pending);e=p.encode(obs)
  with torch.no_grad():
   old=p(e).clone();actions=torch.distributions.Categorical(probs=old).sample().cpu().numpy()
   with p.reference_context():reference=p(e).clone()
  rewards,result=counterfactuals(m,t,cash,q,pending,actions,t+20)
  if str(m.dates[t+20])>='2023-01-01':raise AssertionError('Training reward crossed boundary')
  prob=p(e);identity=float(abs(prob.detach()-old).max());assert identity<2e-6
  loss,stats=objective(prob,old,reference,rewards)
  if not torch.isfinite(loss):raise AssertionError('Nonfinite RL update')
  opt.zero_grad(set_to_none=True);loss.backward();norm=float(torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True));opt.step();opt.zero_grad(set_to_none=True)
  account={'t':t+20,'capital':capital,'cash':result['cash'],'qty':result['qty'].tolist(),'pending':result['pending'].tolist(),'decision':account['decision']+1};state['account']=account if account['decision']<3 else None;state['rl_step']=step+1
  receipt={'step':step+1,'signal_date':obs['asof'],'reward_end':str(m.dates[t+20]),'capital':capital,'actions':actions.tolist(),'prompt_sha256':e['prompt_sha256'],'reward_matrix':rewards.tolist(),'realized_return':float(result['values'][-1]/obs['equity']-1),'trade_shares':result['trade_shares'].tolist(),'likelihood_error':identity,'gradient_norm':norm,**stats};write(out/f'rl_decision_{step+1}.json',receipt)
  if (step+1)%16==0 or step==0:
   save_recovery(out,p,opt,state,rng);status={'state':'counterfactual_rl','seed':seed,'pid':os.getpid(),'step':step+1,'total':args.rl_steps,'step_seconds':time.perf_counter()-begin,'likelihood_error':identity,**stats};write(root/'status.json',status);print(json.dumps(status),flush=True)
  if (step+1)%128==0 or step+1==args.rl_steps:checkpoint(f'rl_{step+1}','rl');save_recovery(out,p,opt,state,rng)
 selected=max(candidates,key=lambda c:c['score']);write(out/'selection.json',selected);load(p,out/selected['checkpoint']);score,reproduced=validate(p,m);original=json.loads((out/f"{selected['checkpoint']}_validation.json").read_text());error=max(float(abs(np.array(original[y]['daily_equity'])-np.array(reproduced[y]['daily_equity'])).max()) for y in original);assert error<.01;write(out/'selected_reload.json',{'max_equity_difference':error});write(out/'selected_sensitivity.json',probe(p,m))
 done={'seed':seed,'state':'trained','selected':selected};write(out/'training_done.json',done);return done

def bootstrap(a,b,initial,seed=7):
 # Descriptive block uncertainty, not correction for adaptive research selection.
 a=np.r_[initial,a];b=np.r_[initial,b];difference=np.log(a[1:]/a[:-1])-np.log(b[1:]/b[:-1]);rng=np.random.default_rng(seed);n=len(difference);samples=[]
 for _ in range(2000):
  starts=rng.integers(0,n,int(np.ceil(n/20)));ix=np.concatenate([(s+np.arange(20))%n for s in starts])[:n];samples.append(difference[ix].sum())
 return {'log_growth_difference':float(difference.sum()),'block20_percentile95':np.quantile(samples,[.025,.975]).tolist(),'caveat':'Descriptive circular block bootstrap; not independent confirmation or multiple-search adjustment.'}

def evaluate_seed(root,m,seed,kind,protocol):
 out=root/f'seed{seed}';done=json.loads((out/'training_done.json').read_text())
 if done['state']!='trained':return
 torch.manual_seed(seed);p=NativePolicy(BASE);load(p,out/'supervised');p.freeze_reference();load(p,out/done['selected']['checkpoint']);metrics=[];primary={}
 for interval,start,end,capitals in [('primary','2025-01-01','2026-01-30',[500,5000,50000,100000,200000]),('year2025','2025-01-01','2025-12-31',[100000]),('jan2026','2026-01-01','2026-01-30',[100000])]:
  for capital in capitals:
   for name in ['selected','untrained','supervised','buy_hold','equal_rebalanced','inverse_vol','momentum','teacher','cash']:
    if name=='untrained':
     with p.model.disable_adapter():r=evaluate(m,p.choose,start,end,capital)
    elif name=='supervised':
     with p.reference_context():r=evaluate(m,p.choose,start,end,capital)
    else:r=evaluate(m,p.choose if name=='selected' else (lambda o:(teacher(o,kind),{'teacher':kind})) if name=='teacher' else baseline(name),start,end,capital)
    write(out/f'test_{interval}_{capital}_{name}.json',r);row={'interval':interval,'capital':capital,'policy':name,**r['metrics']};metrics.append(row);write(out/'test_metrics.json',metrics)
    if interval=='primary' and capital==100000:primary[name]=r
    write(root/'status.json',{'state':'evaluation','seed':seed,'pid':os.getpid(),'latest':row})
 # Isolate added numeric bridge from trained LoRA. Neither is a retrained control.
 p.ablate_numeric=True;r=evaluate(m,p.choose,'2025-01-01','2026-01-30',100000);write(out/'ablation_without_numeric.json',r);p.ablate_numeric=False
 p.bridge_without_lora=True
 with p.model.disable_adapter():r=evaluate(m,p.choose,'2025-01-01','2026-01-30',100000)
 write(out/'ablation_without_lora.json',r);p.bridge_without_lora=False
 write(out/'uncertainty.json',{name:bootstrap(primary['selected']['daily_equity'],primary[name]['daily_equity'],100000) for name in ['untrained','supervised','buy_hold','equal_rebalanced']})
 # Same selected policy under doubled transaction costs; never used for selection.
 from . import market
 original_fee,original_slip=market.FEE,market.SLIPPAGE
 try:
  market.FEE*=2;market.SLIPPAGE*=2
  stress={}
  for name in ['selected','buy_hold','equal_rebalanced']:
   r=evaluate(m,p.choose if name=='selected' else baseline(name),'2025-01-01','2026-01-30',100000);stress[name]=r['metrics'];write(out/f'stress2x_{name}.json',r)
  write(out/'stress2x_summary.json',stress)
 finally:market.FEE,market.SLIPPAGE=original_fee,original_slip
 write(out/'evaluation_done.json',{'state':'completed','metrics_rows':len(metrics)})

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);ap.add_argument('--sft-steps',type=int,default=1024);ap.add_argument('--rl-steps',type=int,default=1536);ap.add_argument('--resume',action='store_true');args=ap.parse_args();root=Path(args.run).resolve();root.mkdir(parents=True,exist_ok=args.resume);torch.set_num_threads(2)
 if os.name=='nt':
  import ctypes
  ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
 try:
  m=Market();audit=select_teacher(m);write(root/'teacher_audit.json',audit);kind=audit['selected'];seeds=[42,123,456]
  protocol={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'args':vars(args),'seeds':seeds,'original_data_sha256':m.manifest['original_sha256'],'universe':m.tickers,'teacher':audit,
   'method':'Causal score SFT then exact per-stock counterfactual conditional policy gradient;20-session log growth minus0.1drawdown. Sample other48 scores, enumerate all7 own scores in original simulator; expected categorical reward minus0.02KL to SFT; reward scale0.01. No learned critic; not long-horizon optimal control.',
   'training':'SFT2018July-2021; learning gate2022; teacher selected2019-2022; RL2018July-2022 with3 consecutive20-session decisions per account; no post2022 reward access.',
   'selection':'SFT selected by qualifying2022 allocation error after fixed1024examples; RL1536updates fixed, validation every128; mean annual net log growth across2023 and2024; untrained/SFT eligible; earliest tie. Three seeds all reported; no cherry-picking seed on test.',
   'test':'2025-Jan2026 primary at100000INR;500,5000,50000,200000 capital diagnostics;2025 andJan2026 slices; deferred until ALL3 seeds finish training. Retrospective, repeatedly accessed in past work.',
   'actions':'49 explicit relative scores0/.25/.5/1/2/4/8, capped proportional projection; at most10percent per asset; fewer than10 nonzero scores intentionally imply cash. Greedy categorical evaluation, stochastic training: mismatch disclosed.',
   'execution':'Whole shares, next-open,10bps fee+5bps slippage, original corporate-action ledger. Integer rounding minimizes squared stock+cash tracking error; zero-target assets not bought;10percent post-cost cap. Changes apply to every policy/baseline. Small capitals may be infeasible.',
   'representation':'Same pinned InternVL3-2B;25->128->1536 input bridge, native-vocabulary/attentionLoRA rank16; all49 stock/account records. Canonical sorting eliminates caller row-order sensitivity but does NOT prove intrinsic permutation-equivariant neural architecture.',
   'controls':'untrained,SFT,teacher,hold,equal rebalance,inversevol,momentum,cash;3seeds; bridge/LoRA ablations; descriptive block-bootstrap;2xcost sensitivity. Separate small numerical policy control uses the same financial counterfactual objective.',
   'limitations':m.audit['assumptions']+['Updated integer allocator and relative score projection supersede previous floor-only allocation; see execution specification above.','Repeated-test adaptive research and survivorship bias','No guarantee of beating baselines','No journal readiness claim before review'],
   'sources':['https://arxiv.org/abs/1705.08926','https://arxiv.org/abs/2102.09361','https://arxiv.org/abs/1706.10059','https://arxiv.org/abs/2608.09880'],
   'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.glob('*.py')}}
  if args.resume:
   prior=json.loads((root/'protocol.json').read_text())
   if prior['source_sha256']!=protocol['source_sha256']:raise ValueError('Resume source mismatch')
  else:
   write(root/'protocol.json',protocol);write(root/'execution_audit.json',m.audit)
   with zipfile.ZipFile(root/'source_snapshot.zip','w',zipfile.ZIP_DEFLATED) as z:
    for p in ROOT.glob('*.py'):z.write(p,p.name)
  for seed in seeds:
   write(root/'status.json',{'state':'loading_seed','seed':seed,'pid':os.getpid()});train_seed(args,root,m,kind,seed,protocol);gc.collect();torch.cuda.empty_cache()
  for seed in seeds:
   if not (root/f'seed{seed}'/'evaluation_done.json').exists():evaluate_seed(root,m,seed,kind,protocol);gc.collect();torch.cuda.empty_cache()
  write(root/'status.json',{'state':'completed','pid':os.getpid(),'seeds':[json.loads((root/f'seed{s}'/'training_done.json').read_text()) for s in seeds],'deployment':'Not promoted. Review results, failure gates, controls and execution assumptions.'})
 except BaseException as exc:write(root/'status.json',{'state':'failed','pid':os.getpid(),'error':repr(exc),'traceback':traceback.format_exc()});raise
 finally:
  if os.name=='nt':ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)

if __name__=='__main__':main()
