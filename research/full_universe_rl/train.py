"""Recoverable all49 SFT -> conditional-learning gate -> financial RL -> evaluation."""
import argparse,copy,datetime,hashlib,json,os,random,time,traceback
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import load_file
from peft import set_peft_model_state_dict
from market import Market,evaluate,baseline,LEVELS
from policy import NativePolicy
from learning import dataset,assess,passes,anchor,CAPITALS
from episodic import WindowSampler,collect,optimize,clipped_loss

ROOT=Path(__file__).resolve().parent

def write(path,data):
 path=Path(path);temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(data,indent=2),encoding='utf-8');temp.replace(path)

def load(policy,path):
 path=Path(path);manifest=json.loads((path/'manifest.json').read_text())
 for name,digest in manifest['sha256'].items():
  if hashlib.sha256((path/name).read_bytes()).hexdigest()!=digest:raise ValueError('Checkpoint hash mismatch')
 result=set_peft_model_state_dict(policy.model,load_file(str(path/'finance_lora/finance/adapter_model.safetensors')),adapter_name='finance')
 if result.unexpected_keys:raise AssertionError(result.unexpected_keys)
 policy.numeric_bridge.load_state_dict(load_file(str(path/'numeric_bridge.safetensors')))

def optimizer(policy,stage):
 rates={'sft':(2e-5,1e-4,1e-3),'rl':(3e-6,1e-5,5e-5)}[stage]
 groups=[[],[],[]]
 for n,p in policy.named_parameters():
  if p.requires_grad:groups[2 if n.startswith('numeric_bridge') else 1 if '.lm_head.' in n else 0].append(p)
 return torch.optim.AdamW([{'params':g,'lr':lr} for g,lr in zip(groups,rates)],weight_decay=0.)

def recovery(out,policy,opt,state,sampler=None):
 payload={'parameters':{n:p.detach().cpu() for n,p in policy.named_parameters() if p.requires_grad},'optimizer':opt.state_dict(),'state':state,'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),'numpy_rng':np.random.get_state(),'python_rng':random.getstate(),'sampler_rng':None if sampler is None else sampler.rng.bit_generator.state,'reference':[p.cpu() for p in getattr(policy,'reference_weights',[])]}
 temp=out/'recovery.tmp';torch.save(payload,temp);temp.replace(out/'recovery.pt')

def sensitivity(policy,m):
 t,_=m.bounds('2024-01-01','2024-12-31');obs=m.observe(t,100000,np.zeros(49,int),100000);a,receipt=policy.choose(obs);result={}
 for kind in ['capital500','capital5000','holdings','reverse_risk','reverse_returns','row_permutation']:
  other=copy.deepcopy(obs)
  if kind.startswith('capital'):
   c=float(kind[7:]);other=m.observe(t,c,np.zeros(49,int),c)
  elif kind=='holdings':other=m.observe(t,10000,np.ones(49,int),100000)
  elif kind=='reverse_risk':
   for r,v in zip(other['stocks'],[r['vol20'] for r in other['stocks']][::-1]):r['vol20']=v
  elif kind=='reverse_returns':
   for r in other['stocks']:
    for k in ['return5','return20','return60','return120']:r[k]*=-1
  with torch.no_grad():p=policy(policy.encode(other,permutation=np.arange(49)[::-1] if kind=='row_permutation' else None)).cpu().numpy()
  result[kind]={'changed_actions':int((a!=p.argmax(1)).sum()),'max_probability_change':float(abs(np.array(receipt['probabilities'])-p).max()),'actions':p.argmax(1).tolist()}
 return {'base_actions':a.tolist(),'probes':result,'interpretation':'Synthetic responsiveness diagnostics, not proof of financial skill. Different capital need not always change actions.'}

def run(args,out):
 started=time.perf_counter();m=Market();write(out/'execution_audit.json',m.audit)
 protocol={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'args':vars(args),'assets':m.tickers,'source_sha256':m.manifest['original_sha256'],
 'training':'SFT July2018-Dec2022; conditional gate2023; RL July2018-Dec2023. Full126-session history for all49 required; no asset removed.',
 'validation':'2024 full year, primary net return at100000INR; earliest checkpoint wins ties; untrained and supervised eligible.',
 'test':'Original dataset only: Jan2025-Jan2026 continuous, plus separately initialized2025 and Jan2026 diagnostics; retrospective previously accessed period, not pristine holdout.',
 'sft':'Inverse-volatility causal teacher with whole-share feasibility and10percent target cap; 25percent synthetic volatility augmentation; all49 targets in every example; class-weighted loss; no future profit labels.',
 'gate':'2023 real observations; balanced accuracy>=.60 and raw+.15; minimum recall>=.20 for classes with>=15 observations; at least4 predicted classes. Stop before RL if gate fails.',
 'rl':'Factorized clipped per-stock ratios, shared60-session own-account return, four trajectories per context, two causal regime-balanced contexts, three20-session decisions. Reward=net log growth minus0.1drawdown. Ratios0.8..1.2, referenceKL0.01, entropy0.003, oldKLstop0.03; 2passes; group std floor.01.',
 'model':'Original pinned InternVL3-2B; rank16 attention/native-vocabulary LoRA plus20->128->1536 numerical input bridge. All49 states precede all49 queries; own state also injected at query. No teacher or external action model at inference.',
 'initialization_caveat':'Untrained means matched system initialization; zero-initialized bridge has not learned numeric encoding. Disabling adapter also disables bridge, so this comparison measures combined adaptation, not LoRA alone. Numerical-only and input-bridge/LoRA ablations required before publication claims.',
 'budget':'10percent requested cap applies at all capitals. Small accounts can remain cash; this is documented discretization, not a trained profit claim. RL samples5000/50000/100000/200000; evaluation includes500.',
 'execution':m.audit['assumptions'],'limitations':['Fixed-universe survivorship bias','Repeated prior retrospective test access','Simplified corporate-action settlement','No claim of global optimum, flawless method, or guaranteed profit'],
 'sources':['https://arxiv.org/abs/2608.09880','https://arxiv.org/abs/1707.06347','https://arxiv.org/abs/2402.03300'],
 'source_files':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.glob('*.py')}}
 if not args.resume:write(out/'protocol.json',protocol)
 else:
  old=json.loads((out/'protocol.json').read_text())
  if old['source_files']!=protocol['source_files']:raise ValueError('Source changed since recovery; explicit migration required')
 write(out/'status.json',{'state':'loading','pid':os.getpid()});policy=NativePolicy(args.base);params=[p for p in policy.parameters() if p.requires_grad]
 rows=dataset(m,args.states,args.seed,True);gate=dataset(m,48,args.seed+77,False)
 encoded=[policy.encode(r['observation'],r['permutation']) for r in rows];ge=[policy.encode(r['observation'],r['permutation']) for r in gate]
 write(out/'input_coverage.json',{'assets':49,'numeric_shape':list(encoded[0]['numeric_state'].shape),'outputs':49,'max_tokens':policy.max_tokens,'truncated':False,'example_text':encoded[0]['text']})
 with torch.no_grad():pristine=policy(encoded[0]).clone()
 candidates=[];sampler=WindowSampler(m,args.seed+1000);state={'phase':'sft','step':0,'iteration':0,'gate_passed':False};opt=optimizer(policy,'sft');gate_history=[]
 def checkpoint(name,stage):
  path=out/name
  if not path.exists():policy.save(path,protocol)
  result=evaluate(m,policy.choose,'2024-01-01','2024-12-31',100000)
  write(out/f'{name}_validation.json',result);row={'checkpoint':name,'stage':stage,'score':result['metrics']['return'],**result['metrics']};candidates.append(row);write(out/'candidates.json',candidates);print(json.dumps({'phase':'validation',**row}),flush=True);return row
 if args.resume:
  r=torch.load(out/'recovery.pt',map_location='cpu',weights_only=False);state=r['state'];opt=optimizer(policy,state['phase'])
  with torch.no_grad():
   for n,p in policy.named_parameters():
    if p.requires_grad:p.copy_(r['parameters'][n].to(p.device))
  opt.load_state_dict(r['optimizer']);torch.set_rng_state(r['torch_rng']);torch.cuda.set_rng_state_all(r['cuda_rng']);np.random.set_state(r['numpy_rng']);random.setstate(r['python_rng'])
  if r['sampler_rng'] is not None:sampler.rng.bit_generator.state=r['sampler_rng']
  if r['reference']:policy.reference_weights=[p.to('cuda') for p in r['reference']]
  candidates=json.loads((out/'candidates.json').read_text());raw_gate=json.loads((out/'raw_gate.json').read_text());gate_history=json.loads((out/'gate_history.json').read_text()) if (out/'gate_history.json').exists() else []
 else:
  write(out/'curriculum.json',rows);write(out/'gate_examples.json',gate)
  raw_gate=assess(policy,gate,ge);write(out/'raw_gate.json',raw_gate);checkpoint('untrained','untrained')
 counts=np.bincount(np.array([r['actions'] for r in rows]).ravel(),minlength=7);weights=1/np.maximum(counts,1)**.5;weights/=weights.mean();class_weights=torch.tensor(weights,dtype=torch.float32,device='cuda');order=np.random.default_rng(args.seed+500).permutation(len(rows));losses=[]
 if state['phase']=='sft':
  for step in range(state['step'],len(rows)):
   index=order[step];p=policy(encoded[index]);labels=torch.tensor(rows[index]['actions'],device=p.device);target=.95*torch.nn.functional.one_hot(labels,7)+.05/7;sample_weight=class_weights[labels];loss=(-(target*p.log()).sum(1)*sample_weight).sum()/sample_weight.sum()
   if not torch.isfinite(loss):raise AssertionError('Nonfinite SFT loss')
   opt.zero_grad(set_to_none=True);loss.backward();norm=float(torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True));opt.step();opt.zero_grad(set_to_none=True);losses.append(float(loss.detach()));state['step']=step+1
   if step==0:
    # Verify an actual update, serialization including bridge, and raw recovery.
    with torch.no_grad():updated=policy(encoded[0]).clone()
    change=float(abs(updated-pristine).max());assert change>1e-8
    policy.save(out/'serialization_probe',{'purpose':'first actual training update'})
    with torch.no_grad():
     for param in params:param.zero_()
    load(policy,out/'serialization_probe')
    with torch.no_grad():
     restored=policy(encoded[0]);error=float(abs(restored-updated).max())
     with policy.model.disable_adapter():raw_error=float(abs(policy(encoded[0])-pristine).max())
    assert error<2e-6 and raw_error<2e-6
    write(out/'update_reload_receipt.json',{'probability_change':change,'reload_error':error,'disabled_recovery_error':raw_error,'frozen_base_gradients_absent':all(p.grad is None for p in policy.parameters() if not p.requires_grad),'peak_gpu_bytes':torch.cuda.max_memory_allocated()})
   if (step+1)%16==0 or step==0:
    recovery(out,policy,opt,state);status={'state':'supervised_training','pid':os.getpid(),'step':step+1,'total':len(rows),'mean_recent_loss':float(np.mean(losses[-16:])),'gradient_norm':norm,'elapsed_seconds':time.perf_counter()-started};write(out/'status.json',status);print(json.dumps(status),flush=True)
   if (step+1)%128==0 or step+1==len(rows):
    score=assess(policy,gate,ge);passed=passes(score,raw_gate);gate_history.append({'step':step+1,'passed':passed,**score});write(out/'gate_history.json',gate_history);print(json.dumps({'phase':'gate',**gate_history[-1]}),flush=True);policy.save(out/f'gate_{step+1}',protocol)
    if passed:state['gate_passed']=True;break
  if not state['gate_passed']:
   recovery(out,policy,opt,state);write(out/'status.json',{'state':'stopped_learning_gate_failed','pid':os.getpid(),'step':state['step'],'gate':gate_history[-1],'elapsed_seconds':time.perf_counter()-started});return
  checkpoint('supervised','supervised');write(out/'supervised_sensitivity.json',sensitivity(policy,m));policy.freeze_reference();state['phase']='rl';opt=optimizer(policy,'rl');recovery(out,policy,opt,state,sampler)
 for iteration in range(state['iteration']+1,args.iterations+1):
  begin=time.perf_counter();buffer,episodes,collection=collect(policy,m,sampler,iteration);stats=optimize(policy,opt,buffer,sampler.rng);state['iteration']=iteration
  write(out/f'episodes_{iteration}.json',episodes)
  if iteration in [1,2,4] or iteration%8==0 or iteration==args.iterations:checkpoint(f'rl_{iteration}','rl')
  recovery(out,policy,opt,state,sampler);status={'state':'financial_rl','pid':os.getpid(),'iteration':iteration,'iterations':args.iterations,'iteration_seconds':time.perf_counter()-begin,'elapsed_seconds':time.perf_counter()-started,**stats,**collection};write(out/'status.json',status);print(json.dumps(status),flush=True)
 selected=max(candidates,key=lambda c:c['score']);write(out/'selection.json',selected);load(policy,out/selected['checkpoint'])
 reproduced=evaluate(m,policy.choose,'2024-01-01','2024-12-31',100000);saved=json.loads((out/f"{selected['checkpoint']}_validation.json").read_text());error=float(abs(np.array(reproduced['daily_equity'])-np.array(saved['daily_equity'])).max());assert error<.01;write(out/'selected_reload.json',{'max_equity_error':error});write(out/'selected_sensitivity.json',sensitivity(policy,m))
 intervals={'primary':['2025-01-01','2026-01-30'],'year2025':['2025-01-01','2025-12-31'],'jan2026':['2026-01-01','2026-01-30']};metrics=[]
 for interval,(start,end) in intervals.items():
  for capital in CAPITALS if interval=='primary' else [100000.]:
   for name in ['selected','untrained','supervised_only','anchor','buy_hold','equal_rebalanced','inverse_vol','momentum','cash']:
    if name=='untrained':
     with policy.model.disable_adapter():result=evaluate(m,policy.choose,start,end,capital)
    elif name=='supervised_only':
     with policy.reference_context():result=evaluate(m,policy.choose,start,end,capital)
    else:result=evaluate(m,policy.choose if name=='selected' else (lambda obs:(anchor(obs),{'teacher':True})) if name=='anchor' else baseline(name),start,end,capital)
    write(out/f'test_{interval}_{int(capital)}_{name}.json',result);row={'interval':interval,'capital':capital,'policy':name,**result['metrics']};metrics.append(row);write(out/'test_metrics.json',metrics);print(json.dumps({'phase':'test',**row}),flush=True)
 write(out/'status.json',{'state':'completed','pid':os.getpid(),'selected':selected,'elapsed_seconds':time.perf_counter()-started,'publication_status':'Requires separate numerical-only/bridge/LoRA controls, uncertainty and repeat seeds; not auto-promoted to demo.'})

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--run',required=True);parser.add_argument('--base',default='C:/Users/Parth/Documents/PolyvalentFinance/local_models/InternVL3-2B');parser.add_argument('--states',type=int,default=1024);parser.add_argument('--iterations',type=int,default=96);parser.add_argument('--seed',type=int,default=42);parser.add_argument('--resume',action='store_true');args=parser.parse_args();out=Path(args.run).resolve();out.mkdir(parents=True,exist_ok=args.resume)
 random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed);torch.cuda.manual_seed_all(args.seed);torch.set_num_threads(2)
 if os.name=='nt':
  import ctypes
  ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
 try:run(args,out)
 except BaseException as exc:write(out/'status.json',{'state':'failed','pid':os.getpid(),'error':repr(exc),'traceback':traceback.format_exc()});raise
 finally:
  if os.name=='nt':ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)

if __name__=='__main__':main()
