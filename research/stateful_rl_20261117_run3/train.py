"""Causal allocation curriculum followed by on-policy episodic financial RL.

The original InternVL vocabulary emits actions. No external trading policy runs
at inference. Supervised-only is explicitly retained as an ablation.
"""
import argparse,copy,datetime,hashlib,json,os,random,time,traceback
from pathlib import Path
import numpy as np
import torch
from peft import set_peft_model_state_dict
from safetensors.torch import load_file
from market import Market,evaluate,baseline,execute
from policy import NativePolicy
from anchor import anchor,audit
from episodic import WindowSampler,collect,optimize

ROOT=Path(__file__).resolve().parent
CAPITALS=[500.,5000.,50000.,100000.,200000.]
INTERVALS={'primary_continuous':['2025-01-01','2026-11-13'],'historical_comparison':['2025-01-01','2026-01-30'],'later_extension':['2026-02-01','2026-11-13']}

def write(path,data):
 path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(data,indent=2),encoding='utf-8');tmp.replace(path)

def load(policy,path):
 result=set_peft_model_state_dict(policy.model,load_file(str(Path(path)/'finance_lora/finance/adapter_model.safetensors')),adapter_name='finance')
 if result.unexpected_keys:raise AssertionError(result.unexpected_keys)

def optimizer_for(policy,attention,head):
 return torch.optim.AdamW([{'params':[p for n,p in policy.named_parameters() if p.requires_grad and '.lm_head.' not in n],'lr':attention},{'params':[p for n,p in policy.named_parameters() if p.requires_grad and '.lm_head.' in n],'lr':head}],weight_decay=0.)

def curriculum(m,n,seed,kind):
 rng=np.random.default_rng(seed);first,last=m.bounds('2015-01-01','2023-12-31');rows=[]
 for i in range(n):
  t=int(rng.integers(first,last+1));capital=float(rng.choice(CAPITALS));w=np.zeros(5) if i%3==0 else rng.dirichlet(np.ones(6))[:5]
  cash,q,_,_=execute(capital,np.zeros(5,int),m.close[t],w);obs=m.observe(t,float(cash[0]),q[0],capital)
  rows.append({'observation':obs,'actions':anchor(obs,kind).tolist(),'permutation':rng.permutation(5).tolist()})
 return rows

def validate(policy,m,out,step,stage):
 result=evaluate(m,policy.choose,'2024-01-01','2024-12-31',100000.)
 write(out/f'validation_step{step}.json',result)
 return {'step':step,'stage':stage,'score':float(np.log1p(result['metrics']['return'])),**result['metrics']}

def sensitivity(policy,m):
 first,last=m.bounds('2024-01-01','2024-12-31');rows=[]
 for t in np.linspace(first,last,6).astype(int):
  obs=m.observe(t,100000.,np.zeros(5,int),100000.);a,receipt=policy.choose(obs)
  probes={}
  for kind in ['reverse_returns','reverse_volatility','small_account','held_inventory','row_permutation']:
   other=copy.deepcopy(obs)
   if kind=='reverse_returns':
    for s in other['stocks']:
     for key in ['return5','return20','return60','return120']:s[key]*=-1
   elif kind=='reverse_volatility':
    for s,v in zip(other['stocks'],[s['vol20'] for s in obs['stocks']][::-1]):s['vol20']=v
   elif kind=='small_account':other=m.observe(t,500.,np.zeros(5,int),500.)
   elif kind=='held_inventory':other=m.observe(t,1000.,np.array([10,10,10,10,10]),100000.)
   if kind=='row_permutation':
    with torch.inference_mode():p=policy(policy.encode(other,permutation=[4,2,0,3,1])).double().cpu().numpy()
    b=p.argmax(1)
   else:b,r=policy.choose(other);p=np.asarray(r['probabilities'])
   probes[kind]={'actions':b.tolist(),'changed_actions':int(np.sum(a!=b)),'max_probability_change':float(abs(np.asarray(receipt['probabilities'])-p).max())}
  rows.append({'date':obs['asof'],'actions':a.tolist(),'probes':probes})
 return {'scope':'Validation counterfactual diagnostic, not profit or causal-interpretation proof. Row-permutation differences are an invariance limitation.','cases':rows}

def smoke(args,out):
 m=Market();policy=NativePolicy(args.base);sampler=WindowSampler(m,args.seed)
 obs=m.observe(sampler.buckets[sampler.keys[0]][0],100000.,np.zeros(5,int),100000.);e=policy.encode(obs)
 with torch.inference_mode():
  original=policy(e).detach().clone()
  with policy.model.disable_adapter():raw=policy(e).detach().clone()
 assert float(abs(original-raw).max())<2e-6
 start=time.perf_counter();buffer,episodes,collection=collect(policy,m,sampler,0,groups=1)
 policy.freeze_reference();opt=optimizer_for(policy,1e-6,1e-5);before=[p.detach().clone() for p in policy.parameters() if p.requires_grad]
 stats=optimize(policy,opt,buffer,sampler.rng)
 change=max(float(abs(p-b).max()) for p,b in zip([p for p in policy.parameters() if p.requires_grad],before));assert change>0
 assert all(p.grad is None for n,p in policy.named_parameters() if 'lora_' not in n)
 with torch.inference_mode():
  updated=policy(e).detach().clone()
  with policy.reference_context():ref=policy(e).detach().clone()
 assert float(abs(ref-original).max())<2e-6
 policy.save(out/'serialization_probe',{'purpose':'Preflight, not candidate'})
 with torch.no_grad():
  for p in policy.parameters():
   if p.requires_grad:p.zero_()
 load(policy,out/'serialization_probe')
 with torch.inference_mode():
  restored=policy(e).detach().clone()
  with policy.model.disable_adapter():disabled=policy(e).detach().clone()
 reload_error=float(abs(restored-updated).max());recovery=float(abs(disabled-original).max());assert max(reload_error,recovery)<2e-6
 receipt={'status':'passed','parameter_change':change,'reload_difference':reload_error,'disabled_recovery_difference':recovery,'reference_swap_difference':float(abs(ref-original).max()),'elapsed_seconds':time.perf_counter()-start,'peak_gpu_bytes':torch.cuda.max_memory_allocated(),'tokens':policy.max_tokens,'optimizer':stats,'collection':collection,'episodes':episodes}
 write(out/'smoke.json',receipt);print(json.dumps({k:v for k,v in receipt.items() if k!='episodes'}),flush=True)

def train(args,out):
 started=time.perf_counter();m=Market();teacher=audit(out/'anchor_audit.json');kind=teacher['selected']
 protocol={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'arguments':vars(args),'algorithm':'causal-SFT then on-policy clipped group-relative episodic financial policy gradient','train':'2015-2023','validation':'2024 net growth at100000 INR, earliest tie; zero and supervised-only eligible','test_intervals':INTERVALS,'test_caveat':'Retrospective evaluation repeatedly seen by project; method selection is adaptive, not untouched confirmation. Fixed universe has survivorship/selection limitations.',
 'teacher':kind,'teacher_selection':teacher['criterion'],'SFT':'768 causal random valid account states, two passes, label smoothing0.05, shuffled rows; no future-return targets','RL':'Four60-session own-account trajectories per context, two contexts/update, three20-session decisions per episode; 128updates. Terminal net log growth minus0.1 drawdown. Group centered/std-scaled advantage with0.01floor; joint five-action probability ratio clipped0.8..1.2, two update passes, referenceKL0.01, entropy0.003, oldKL stop0.05. Reference is validation-selected supervised checkpoint.',
 'sampling':'Past-only up/down60session trend and high/low20session volatility buckets balanced; no future-regime sampling','execution':'Whole shares, no shorts/leverage, next-open fills,0.10% fee+0.05%slippage each side;35%stock/60%sector caps with affordability relaxation; dividends creditedexdate; splits reconstructed; no tax/marketimpact model.',
 'model':'Pinned InternVL3-2B native attention and original-vocabulary LoRA rank16, no numerical replacement head; five conditionally independent categorical action slots, full account/stock input, neutral IDqueries with shuffled rows; greedy evaluation vs stochastic training mismatch explicitly retained.',
 'capital_probes':CAPITALS,'selection_limit':'No claim of optimum, flawless process, independent unseen test, journal acceptance or universal outperformance. RL gain must be measured separately from SFT.',
 'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.glob('*.py')},'data_manifest_sha256':hashlib.sha256((ROOT.parent/'stateful_rl_20261117/data_v2/manifest.json').read_bytes()).hexdigest()}
 write(out/'protocol.json',protocol);write(out/'status.json',{'state':'loading','pid':os.getpid()});policy=NativePolicy(args.base)
 candidates=[]
 def checkpoint(step,stage):
  policy.save(out/f'checkpoint_{step}',protocol);v=validate(policy,m,out,step,stage);candidates.append(v);write(out/'candidates.json',candidates);print(json.dumps({'phase':'validation',**v}),flush=True);return v
 checkpoint(0,'untrained');rows=curriculum(m,args.states,args.seed,kind);write(out/'curriculum.json',rows)
 encoded=[policy.encode(r['observation'],r['permutation']) for r in rows]
 write(out/'input_coverage.json',{'tokens_max':policy.max_tokens,'example':encoded[0]['text'],'truncated':False})
 with torch.inference_mode():pristine=policy(encoded[0]).detach().clone()
 opt=optimizer_for(policy,1e-5,5e-5);params=[p for p in policy.parameters() if p.requires_grad];rng=np.random.default_rng(args.seed+500);step=0
 for epoch in range(2):
  opt.zero_grad(set_to_none=True);losses=[];correct=[]
  for j,index in enumerate(rng.permutation(len(rows))):
   p=policy(encoded[index]);labels=torch.tensor(rows[index]['actions'],device=p.device);target=.95*torch.nn.functional.one_hot(labels,7)+.05/7
   loss=-(target*p.log()).sum(-1).mean();assert torch.isfinite(loss);(loss/4).backward();losses.append(float(loss.detach()));correct.append(float((p.argmax(1)==labels).float().mean()))
   if (j+1)%4==0 or j+1==len(rows):torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True);opt.step();opt.zero_grad(set_to_none=True)
   step+=1
   if step%32==0:
    status={'state':'supervised_warmup','pid':os.getpid(),'step':step,'total':2*len(rows),'mean_recent_loss':float(np.mean(losses[-32:])),'recent_action_accuracy':float(np.mean(correct[-32:])),'elapsed_seconds':time.perf_counter()-started};write(out/'status.json',status);print(json.dumps(status),flush=True)
   if step%384==0 or j+1==len(rows):checkpoint(step,'supervised')
 sft=max((c for c in candidates if c['stage']=='supervised'),key=lambda c:(c['score'],-c['step']));write(out/'supervised_selection.json',sft);load(policy,out/f'checkpoint_{sft["step"]}');policy.freeze_reference()
 opt=optimizer_for(policy,1e-6,1e-5);sampler=WindowSampler(m,args.seed+1000);best=max(c['score'] for c in candidates);stale=0
 for iteration in range(1,args.iterations+1):
  begin=time.perf_counter();buffer,episodes,collection=collect(policy,m,sampler,iteration,groups=2);stats=optimize(policy,opt,buffer,sampler.rng)
  write(out/f'episodes_{iteration}.json',episodes);status={'state':'episodic_rl','pid':os.getpid(),'iteration':iteration,'iterations':args.iterations,'elapsed_seconds':time.perf_counter()-started,'iteration_seconds':time.perf_counter()-begin,**stats,**collection};write(out/'status.json',status);print(json.dumps(status),flush=True)
  if iteration in [1,2,4] or iteration%8==0 or iteration==args.iterations:
   v=checkpoint(2000+iteration,'rl')
   if v['score']>best+1e-10:best=v['score'];stale=0
   else:stale+=1
   if stale>=12:break
 selected=max(candidates,key=lambda c:(c['score'],-c['step']));write(out/'selection.json',selected);load(policy,out/f'checkpoint_{selected["step"]}')
 check=evaluate(m,policy.choose,'2024-01-01','2024-12-31',100000.);saved=json.loads((out/f'validation_step{selected["step"]}.json').read_text());error=float(abs(np.asarray(check['daily_equity'])-np.asarray(saved['daily_equity'])).max());assert error<.01
 with policy.model.disable_adapter(),torch.inference_mode():recovery=float(abs(policy(encoded[0])-pristine).max())
 assert recovery<2e-6;write(out/'selected_reload_verification.json',{'max_equity_difference':error,'disabled_recovery_difference':recovery});write(out/'selected_state_sensitivity.json',sensitivity(policy,m))
 metrics=[];policies=['trained','untrained','supervised_only','anchor','buy_hold','equal_rebalanced','inverse_vol','momentum','cash']+[f'single_hold_{i}' for i in range(5)]
 for interval,(start,end) in INTERVALS.items():
  for capital in CAPITALS:
   for name in policies:
    if name=='untrained':
     with policy.model.disable_adapter():result=evaluate(m,policy.choose,start,end,capital)
    elif name=='supervised_only':
     with policy.reference_context():result=evaluate(m,policy.choose,start,end,capital)
    else:result=evaluate(m,policy.choose if name=='trained' else (lambda o:(anchor(o,kind),{'anchor':kind})) if name=='anchor' else baseline(name),start,end,capital)
    write(out/f'test_{interval}_{int(capital)}_{name}.json',result)
    row={'interval':interval,'capital':capital,'policy':name,'unique_action_vectors':len({tuple(d['actions']) for d in result['decisions'] if d['actions'] is not None}),**result['metrics']};metrics.append(row);write(out/'test_metrics.json',metrics);print(json.dumps({'phase':'test',**row}),flush=True)
 comparisons=[]
 for interval in INTERVALS:
  for capital in CAPITALS:
   group={r['policy']:r for r in metrics if r['interval']==interval and r['capital']==capital};tr=group['trained']['return'];comparisons.append({'interval':interval,'capital':capital,'trained_return':tr,'selected_stage':selected['stage'],'positive_and_beats_raw_and_hold':tr>max(0,group['untrained']['return'],group['buy_hold']['return']),**{f'trained_minus_{name}':tr-group[name]['return'] for name in ['untrained','supervised_only','anchor','buy_hold','equal_rebalanced']}})
 write(out/'comparison_summary.json',comparisons);write(out/'status.json',{'state':'completed','pid':os.getpid(),'selected':selected,'elapsed_seconds':time.perf_counter()-started,'comparisons':comparisons,'deployment':'Not automatically promoted; require correctness/state-response/benchmark review.'})

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=['smoke','train'],required=True);parser.add_argument('--run',required=True);parser.add_argument('--base',required=True);parser.add_argument('--states',type=int,default=768);parser.add_argument('--iterations',type=int,default=128);parser.add_argument('--seed',type=int,default=42);args=parser.parse_args()
 out=Path(args.run).resolve();out.mkdir(parents=True,exist_ok=False);random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed);torch.cuda.manual_seed_all(args.seed);torch.set_num_threads(2)
 if os.name=='nt':
  import ctypes
  ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
 try:(smoke if args.mode=='smoke' else train)(args,out)
 except BaseException as exc:write(out/'status.json',{'state':'failed','pid':os.getpid(),'error':repr(exc),'traceback':traceback.format_exc()});raise
 finally:
  if os.name=='nt':ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)

if __name__=='__main__':main()
