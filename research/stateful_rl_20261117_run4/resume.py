"""Shutdown recovery: weights-only warm restart when no optimizer exists.
Full recovery snapshots are saved after every new RL iteration.
"""
from train import *
import shutil

def resume(args,out):
 source=Path(args.source);old_protocol=json.loads((source/'protocol.json').read_text());old_status=json.loads((source/'status.json').read_text())
 candidates=json.loads((source/'candidates.json').read_text());latest=max(candidates,key=lambda c:c['step']);start_iteration=latest['step']-10000
 recovery_path=source/'training_state_latest.pt';recovery=None
 if recovery_path.exists():
  recovery=torch.load(recovery_path,map_location='cpu',weights_only=False);start_iteration=recovery['iteration']
  candidates=[c for c in candidates if c['step']<=10000+start_iteration];latest=max(candidates,key=lambda c:c['step'])
 for p in source.glob('checkpoint_*'):
  if p.is_dir():
   if int(p.name.rsplit('_',1)[1])>10000+start_iteration:continue
   manifest=json.loads((p/'manifest.json').read_text())['sha256']
   for name,digest in manifest.items():
    if hashlib.sha256((p/name).read_bytes()).hexdigest()!=digest:raise AssertionError('Checkpoint hash mismatch '+str(p/name))
   shutil.copytree(p,out/p.name)
 for pattern in ['validation_step*.json','curriculum.json','supervised_selection.json','learning_gate*.json','raw_learning_gate.json','dataset_summary.json']:
  for p in source.glob(pattern):shutil.copy2(p,out/p.name)
 if recovery is not None:shutil.copytree(source/recovery['checkpoint'],out/recovery['checkpoint'])
 protocol=dict(old_protocol);protocol['recovery']={'source':str(source),'last_reported_iteration':old_status.get('iteration'),'resumed_after_iteration':start_iteration,'optimizer_restored':recovery is not None,'method':'exact saved optimizer/RNG recovery' if recovery else 'weights-only warm restart; original optimizer/RNG were not persisted','lost_reported_iterations':max(0,old_status.get('iteration',start_iteration)-start_iteration),'new_sampler_seed':4242 if recovery is None else None,'recovery_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
 write(out/'protocol.json',protocol);write(out/'candidates.json',candidates);write(out/'status.json',{'state':'loading_recovered_weights','pid':os.getpid(),**protocol['recovery']})
 started=time.perf_counter();m=Market();policy=NativePolicy(args.base);kind='inverse_vol'
 sft=json.loads((source/'supervised_selection.json').read_text());load(policy,out/f'checkpoint_{sft["step"]}');policy.freeze_reference()
 load(policy,out/(recovery['checkpoint'] if recovery is not None else f'checkpoint_{latest["step"]}'))
 if recovery is None:
  observed=evaluate(m,policy.choose,'2024-01-01','2024-12-31',100000.);saved=json.loads((out/f'validation_step{latest["step"]}.json').read_text());error=float(abs(np.asarray(observed['daily_equity'])-np.asarray(saved['daily_equity'])).max());assert error<.01
  write(out/'resume_reload_verification.json',{'checkpoint':latest['step'],'maximum_equity_difference':error})
 rows=json.loads((out/'curriculum.json').read_text());encoded=[policy.encode(rows[0]['observation'],rows[0]['permutation'],focus=rows[0]['focus'])]
 with policy.model.disable_adapter(),torch.inference_mode():pristine=policy(encoded[0]).detach().clone()
 opt=optimizer_for(policy,3e-6,3e-5);sampler=WindowSampler(m,4242);best=max(c['score'] for c in candidates);stale=0
 if recovery is not None:
  opt.load_state_dict(recovery['optimizer']);sampler.rng.bit_generator.state=recovery['sampler_rng'];torch.set_rng_state(recovery['torch_rng']);torch.cuda.set_rng_state_all(recovery['cuda_rng']);np.random.set_state(recovery['numpy_rng']);random.setstate(recovery['python_rng']);best=recovery['best'];stale=recovery['stale']
 for iteration in range(start_iteration+1,args.iterations+1):
  begin=time.perf_counter();buffer,episodes,collection=collect(policy,m,sampler,iteration,groups=2);stats=optimize(policy,opt,buffer,sampler.rng)
  write(out/f'episodes_{iteration}.json',episodes)
  if iteration%8==0 or iteration==args.iterations:
   step=10000+iteration;policy.save(out/f'checkpoint_{step}',protocol);v=validate(policy,m,out,step,'rl_resumed');candidates.append(v);write(out/'candidates.json',candidates);print(json.dumps({'phase':'validation',**v}),flush=True)
   if v['score']>best+1e-10:best=v['score'];stale=0
   else:stale+=1
  checkpoint=f'recovery_checkpoint_{iteration}';policy.save(out/checkpoint,protocol)
  state={'iteration':iteration,'checkpoint':checkpoint,'optimizer':opt.state_dict(),'sampler_rng':sampler.rng.bit_generator.state,'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),'numpy_rng':np.random.get_state(),'python_rng':random.getstate(),'best':best,'stale':stale}
  temp=out/'training_state_latest.tmp';torch.save(state,temp);temp.replace(out/'training_state_latest.pt')
  status={'state':'episodic_rl_resumed','pid':os.getpid(),'iteration':iteration,'iterations':args.iterations,'elapsed_seconds':time.perf_counter()-started,'iteration_seconds':time.perf_counter()-begin,'full_recovery_saved':True,**stats,**collection};write(out/'status.json',status);print(json.dumps(status),flush=True)
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
 parser=argparse.ArgumentParser();parser.add_argument('--source',required=True);parser.add_argument('--run',required=True);parser.add_argument('--base',required=True);parser.add_argument('--iterations',type=int,default=96);args=parser.parse_args()
 out=Path(args.run).resolve();out.mkdir(parents=True,exist_ok=False);random.seed(4242);np.random.seed(4242);torch.manual_seed(4242);torch.cuda.manual_seed_all(4242);torch.set_num_threads(2)
 import ctypes
 ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
 try:resume(args,out)
 except BaseException as exc:write(out/'status.json',{'state':'failed','pid':os.getpid(),'error':repr(exc),'traceback':traceback.format_exc()});raise
 finally:ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)

if __name__=='__main__':main()
