"""Exact all-action, one-rebalance-horizon financial policy-gradient training.

This is offline contextual financial RL, not PPO, teacher SFT, or a claim of
optimal long-horizon credit assignment. Future prices appear only in rewards.
"""
import argparse,copy,datetime,hashlib,json,os,random,sys,time,traceback
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import load_file
from peft import set_peft_model_state_dict
from market import Market,evaluate,baseline,ACTIONS
from policy import NativePolicy,expected_reward

ROOT=Path(__file__).resolve().parent
CAPITALS=[500.,50000.,100000.,200000.]

def write(path,data):
 path=Path(path);temporary=path.with_suffix(path.suffix+'.tmp')
 temporary.write_text(json.dumps(data,indent=2),encoding='utf-8');temporary.replace(path)

def seed(value):
 random.seed(value);np.random.seed(value);torch.manual_seed(value);torch.cuda.manual_seed_all(value);torch.set_num_threads(2)

def scalar_loss(policy,encoded,reward,reference):
 p=policy(encoded);r=torch.as_tensor(reward,device='cuda');r=r-r.mean()
 gain=expected_reward(p,r)/.02
 kl=(p*(p.log()-reference.log())).sum(-1).mean()
 return -gain+.02*kl,{'expected_centered_return':float(gain.detach())*.02,'kl':float(kl.detach())}

def sensitivity(policy,market):
 t,_=market.bounds('2021-01-01','2021-06-01');zero=np.zeros(5,int)
 obs=[market.observe(t,500.,zero,500.),market.observe(t,200000.,zero,200000.),
      market.observe(t,1000.,np.array([0,50,0,0,0]),100000.),market.observe(t-60,200000.,zero,200000.)]
 results=[]
 for o in obs:
  a,receipt=policy.choose(o);results.append({'observation':o,'actions':a.tolist(),**receipt})
 p=np.array([r['probabilities'] for r in results]);difference=float(np.max(abs(p[0]-p[1])))
 if difference<1e-7:raise AssertionError('Account-change inputs have no measurable model effect')
 return {'probes':results,'max_probability_change_cash500_vs200000':difference,'unique_action_vectors':len({tuple(r['actions']) for r in results})}

def smoke(args,out):
 m=Market();states=m.sample_states(4,args.seed);start=time.perf_counter();policy=NativePolicy(args.base)
 encoded=[policy.encode(s['observation']) for s in states]
 write(out/'prompt_examples.json',[{'observation':s['observation'],'text':e['text'],'tokens':e['ids'].shape[1],'positions':e['positions']} for s,e in zip(states,encoded)])
 with torch.inference_mode():
  original=policy(encoded[0]).detach()
  with policy.model.disable_adapter():disabled=policy(encoded[0]).detach()
 error=float((original-disabled).abs().max());assert error<2e-6
 trainable=[p for p in policy.parameters() if p.requires_grad]
 times=[];attention_gradient=0.;readout_gradient=0.
 for step in range(3):
  policy.zero_grad(set_to_none=True);torch.cuda.synchronize();began=time.perf_counter()
  with torch.no_grad():reference=policy(encoded[step])
  loss,stats=scalar_loss(policy,encoded[step],states[step]['reward'],reference)
  if not torch.isfinite(loss):raise AssertionError('Nonfinite loss')
  loss.backward();torch.nn.utils.clip_grad_norm_(trainable,1.,error_if_nonfinite=True);torch.cuda.synchronize();times.append(time.perf_counter()-began)
  for name,p in policy.named_parameters():
   if p.requires_grad and p.grad is not None:
    if '.self_attn.' in name:attention_gradient+=float(p.grad.abs().sum())
    if '.lm_head.' in name:readout_gradient+=float(p.grad.abs().sum())
   elif 'lora_' not in name and p.grad is not None:raise AssertionError('Base received a gradient')
 if min(attention_gradient,readout_gradient)<=0:raise AssertionError('LoRA path disconnected')
 optimizer=torch.optim.AdamW([{'params':[p for n,p in policy.named_parameters() if p.requires_grad and '.lm_head.' not in n],'lr':3e-6},{'params':[p for n,p in policy.named_parameters() if p.requires_grad and '.lm_head.' in n],'lr':3e-5}],weight_decay=0.)
 before=[p.detach().clone() for p in trainable];optimizer.step()
 change=max(float((p-b).abs().max()) for p,b in zip(trainable,before));assert change>0
 del before
 with torch.inference_mode():updated=policy(encoded[0]).detach().clone()
 policy.save(out/'serialization_probe',{'purpose':'One optimizer-step export/reload test; not a model-quality candidate'})
 with torch.no_grad():
  for p in trainable:p.zero_()
 result=set_peft_model_state_dict(policy.model,load_file(str(out/'serialization_probe/finance_lora/finance/adapter_model.safetensors')),adapter_name='finance')
 assert not result.unexpected_keys,result.unexpected_keys
 with torch.inference_mode():reloaded=policy(encoded[0]).detach()
 reload_error=float((updated-reloaded).abs().max());assert reload_error<2e-6
 with policy.model.disable_adapter(),torch.inference_mode():recovered=policy(encoded[0]).detach()
 recovery_error=float((original-recovered).abs().max());assert recovery_error<2e-6
 checks=sensitivity(policy,m)
 receipt={'status':'passed','base_disabled_difference':error,'one_step_max_parameter_change':change,'export_reload_difference':reload_error,'post_update_disabled_recovery_difference':recovery_error,'trainable_parameters':sum(p.numel() for p in trainable),'max_tokens':policy.max_tokens,'attention_gradient_abs_sum':attention_gradient,'vocabulary_LoRA_gradient_abs_sum':readout_gradient,'training_pass_seconds_including_uncached_reference':times,'peak_gpu_bytes':torch.cuda.max_memory_allocated(),'load_and_check_seconds':time.perf_counter()-start,'state_sensitivity':checks}
 write(out/'smoke.json',receipt);print(json.dumps({k:v for k,v in receipt.items() if k!='state_sensitivity'},indent=2),flush=True)

def validate(policy,market,out,epoch):
 rows=[]
 for capital in CAPITALS:
  write(out/'status.json',{'state':'validation','pid':os.getpid(),'epoch':epoch,'capital':capital})
  result=evaluate(market,policy.choose,'2022-01-01','2022-12-31',capital)
  write(out/f'validation_epoch{epoch}_capital{int(capital)}.json',result)
  rows.append({'capital':capital,**result['metrics']})
 score=float(np.log1p(next(r['return'] for r in rows if r['capital']==100000.)))
 return {'epoch':epoch,'score':score,'criterion':'validation net wealth growth on primary100000INR; earliest tie; epoch0 eligible; other capitals are prespecified stress checks','rows':rows}

def train(args,out):
 market=Market();states=market.sample_states(args.states,args.seed);write(out/'state_inventory.json',[{k:v for k,v in s.items() if k!='reward'} for s in states])
 np.savez_compressed(out/'counterfactual_training_rewards.npz',rewards=np.array([s['reward'] for s in states]))
 protocol={'algorithm':'state_conditioned_all_action_one_rebalance_financial_policy_gradient','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'arguments':vars(args),
  'train':'2015-01-01 through 2021-12-31; rewards never cross boundary','validation':'2022; checkpoint selection only',
  'test_intervals':{'primary_continuous':['2025-01-01','2026-11-13'],'historical_comparison':['2025-01-01','2026-01-30'],'later_extension':['2026-02-01','2026-11-13']},
  'test_caveat':'Retrospective project history; no blanket claim of untouched confirmation. Different five-stock universe and integer execution mean old13.28% is not a matched comparator.',
  'state_distribution':'Random valid cash/integer inventory states from training dates and fixed training capitals; offline contextual RL, not on-policy long-episode PPO.',
  'objective':'Exact expectation of five-session net log wealth over all7^5 joint per-stock actions, scaled by0.02, plus0.02 KL to pristine policy on the same observation.',
  'execution':'Observe known close; execute next open; integer shares; sell first; proportionally fund rounded buys; .10%fees+.05%slippage per side; dividends/splits before opening trades.',
  'evaluation_capitals':CAPITALS,'selection':'2022 net wealth growth at100000INR; earliest tie; epoch0 eligible','evaluation_actions':'deterministic per-stock argmax; no test-time sampling or cached action fallback','base_id':'OpenGVLab/InternVL3-2B','base_revision':'899155015275a9b7338c7f4677e19c784e0e5a21',
  'trainable':'Only rank16 full-InternVL-root attention and original vocabulary-head LoRA; no replacement numerical policy/head',
  'attention_lr':3e-6,'vocabulary_head_lr':3e-5,'gradient_clip':1.,'weight_decay':0.,'seed':args.seed,
  'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.glob('*.py')},'data_manifest_sha256':hashlib.sha256((ROOT/'data_v2/manifest.json').read_bytes()).hexdigest()}
 write(out/'protocol.json',protocol);write(out/'status.json',{'state':'loading_model','pid':os.getpid()})
 policy=NativePolicy(args.base);encoded=[policy.encode(s['observation']) for s in states]
 write(out/'input_coverage.json',{'states':len(encoded),'min_tokens':min(e['ids'].shape[1] for e in encoded),'max_tokens':max(e['ids'].shape[1] for e in encoded),'truncated':False,'example_prompt':encoded[0]['text']})
 refs=[];started=time.perf_counter()
 with torch.inference_mode():
  for i,e in enumerate(encoded):
   refs.append(policy(e).detach())
   if i%64==0:
    progress={'state':'pristine_reference','pid':os.getpid(),'example':i,'total':len(encoded),'elapsed_seconds':time.perf_counter()-started}
    write(out/'status.json',progress);print(json.dumps(progress),flush=True)
 reference_probes=[refs[i].detach().cpu() for i in range(min(3,len(refs)))]
 groups=[{'params':[p for n,p in policy.named_parameters() if p.requires_grad and '.lm_head.' not in n],'lr':3e-6},
         {'params':[p for n,p in policy.named_parameters() if p.requires_grad and '.lm_head.' in n],'lr':3e-5}]
 optimizer=torch.optim.AdamW(groups,weight_decay=0.);parameters=[p for p in policy.parameters() if p.requires_grad];candidates=[]
 policy.save(out/'checkpoint_0',protocol);v=validate(policy,market,out,0);candidates.append(v);write(out/'candidates.json',candidates)
 for epoch in range(1,args.epochs+1):
  order=np.random.default_rng(args.seed+epoch).permutation(len(states));optimizer.zero_grad(set_to_none=True);losses=[]
  for j,i in enumerate(order):
   loss,stats=scalar_loss(policy,encoded[i],states[i]['reward'],refs[i]);(loss/args.accum).backward();losses.append(float(loss.detach()))
   if (j+1)%args.accum==0 or j==len(order)-1:
    norm=float(torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True));optimizer.step();optimizer.zero_grad(set_to_none=True)
   if (j+1)%16==0:
    state={'state':'training','pid':os.getpid(),'epoch':epoch,'epochs':args.epochs,'example':j+1,'examples_per_epoch':len(order),'mean_loss':float(np.mean(losses[-16:])),'elapsed_seconds':time.perf_counter()-started,'gradient_norm':norm,'max_tokens':policy.max_tokens,**stats}
    write(out/'status.json',state);print(json.dumps(state),flush=True)
  policy.save(out/f'checkpoint_{epoch}',protocol)
  torch.save({'optimizer':optimizer.state_dict(),'epoch':epoch,'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all()},out/f'optimizer_epoch{epoch}.pt')
  v=validate(policy,market,out,epoch);candidates.append(v);write(out/'candidates.json',candidates);print(json.dumps({'phase':'validation',**v}),flush=True)
 selected=max(candidates,key=lambda r:(r['score'],-r['epoch']));write(out/'selection.json',selected)
 weights=out/f'checkpoint_{selected["epoch"]}/finance_lora/finance/adapter_model.safetensors';set_peft_model_state_dict(policy.model,load_file(str(weights)),adapter_name='finance')
 reloaded=evaluate(market,policy.choose,'2022-01-01','2022-12-31',100000.)
 original=json.loads((out/f'validation_epoch{selected["epoch"]}_capital100000.json').read_text())
 reload_error=float(np.max(np.abs(np.array(reloaded['daily_equity'])-np.array(original['daily_equity']))))
 if reload_error>.01:raise AssertionError('Selected saved weights do not reproduce validation path')
 write(out/'selected_reload_verification.json',{'selected_epoch':selected['epoch'],'max_validation_equity_difference':reload_error})
 with policy.model.disable_adapter(),torch.inference_mode():
  errors=[float((policy(encoded[i]).cpu()-reference_probes[i]).abs().max()) for i in range(len(reference_probes))]
 if max(errors)>2e-6:raise AssertionError('Disabled adapter failed to recover pristine outputs')
 write(out/'selected_state_sensitivity.json',sensitivity(policy,market))
 rows=[]
 for interval,(begin,end) in protocol['test_intervals'].items():
  for capital in CAPITALS:
   for name in ['trained','untrained','buy_hold','equal_rebalanced','inverse_vol','momentum','cash',*[f'single_hold_{i}' for i in range(5)]]:
    if name=='untrained':
     with policy.model.disable_adapter():result=evaluate(market,policy.choose,begin,end,capital)
    else:result=evaluate(market,policy.choose if name=='trained' else baseline(name),begin,end,capital)
    file=out/f'test_{interval}_{int(capital)}_{name}.json';write(file,result)
    unique=len({tuple(d['actions']) for d in result['decisions'] if d['actions'] is not None})
    rows.append({'interval':interval,'capital':capital,'policy':name,'unique_model_action_vectors':unique,**result['metrics']});write(out/'test_metrics.json',rows)
    print(json.dumps({'phase':'test',**rows[-1]}),flush=True)
 comparisons=[]
 for interval in protocol['test_intervals']:
  for capital in CAPITALS:
   group={r['policy']:r for r in rows if r['interval']==interval and r['capital']==capital};tr=group['trained']
   comparisons.append({'interval':interval,'capital':capital,'trained_minus_untrained_return':tr['return']-group['untrained']['return'],'trained_minus_equal_holding_return':tr['return']-group['buy_hold']['return'],'trained_minus_best_single_holding_return':tr['return']-max(group[f'single_hold_{i}']['return'] for i in range(5)),'unique_trained_action_vectors':tr['unique_model_action_vectors'],'training_selected':selected['epoch']>0,'beats_both_primary_comparators':tr['return']>max(group['untrained']['return'],group['buy_hold']['return'])})
 write(out/'comparison_summary.json',comparisons)
 write(out/'status.json',{'state':'completed','pid':os.getpid(),'selected_epoch':selected['epoch'],'elapsed_seconds':time.perf_counter()-started,'disabled_adapter_errors':errors,'test_metrics':rows,'comparisons':comparisons,'deployment':'Not automatically promoted. Review correctness, state-response and comparator gates first.'})

def main():
 p=argparse.ArgumentParser();p.add_argument('--mode',choices=['smoke','train'],required=True);p.add_argument('--run',required=True);p.add_argument('--base',required=True);p.add_argument('--epochs',type=int,default=8);p.add_argument('--states',type=int,default=768);p.add_argument('--accum',type=int,default=4);p.add_argument('--seed',type=int,default=42);a=p.parse_args()
 out=Path(a.run).resolve();out.mkdir(parents=True,exist_ok=False);seed(a.seed)
 if os.name=='nt':
  import ctypes
  ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)  # Temporary per-process sleep inhibition; no display wake or schedule.
 try:(smoke if a.mode=='smoke' else train)(a,out)
 except BaseException as exc:
  write(out/'status.json',{'state':'failed','pid':os.getpid(),'error':repr(exc),'traceback':traceback.format_exc()});raise
 finally:
  if os.name=='nt':ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)

if __name__=='__main__':main()
