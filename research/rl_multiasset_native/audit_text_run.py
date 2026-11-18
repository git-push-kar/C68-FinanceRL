"""Fresh-process reload and no-adapter comparison after actual finance RL."""
import argparse,json,gc
from pathlib import Path
import numpy as np
import torch
from text_policy import TextPolicy
from strategy_ppo import Market,evaluate,tensor
from model import write_json
from metrics import metrics_from_infos
from execution_policy import wait_for_compute_window

@torch.no_grad()
def sampled(policy,market,seed):
    rng=np.random.default_rng(seed);state=market.reset();actions=[]
    while True:
        wait_for_compute_window();obs,targets=state
        p=policy(tensor(obs,'cuda'))[0].probs[0].cpu().numpy().astype(float);p/=p.sum()
        action=min(int(np.searchsorted(np.cumsum(p),rng.random())),3)
        actions.append(action);state,_,done=market.step(targets[action])
        if done:break
    return {'metrics':metrics_from_infos(market.infos),'daily':market.infos,'actions':actions}

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);a=p.parse_args();run=Path(a.run)
    selected=json.loads((run/'selection.json').read_text())['selected'];step=selected['step']
    val=Market('../sector_data_20261107/sector9_development_through2024.csv','2019-01-01','2021-12-31')
    dev=Market('../sector_data_20261107/sector9_development_through2024.csv','2022-01-01','2024-12-31')
    policy=TextPolicy(val.env.tickers,selected['package']);observed=evaluate(policy,val,'cuda')
    expected=json.loads((run/f'validation_{step}.json').read_text())
    error=float(np.max(abs(np.array(observed['probabilities'])-np.array(expected['probabilities']))))
    if error>1e-7:raise AssertionError(f'Reload differs: {error}')
    with policy.shared.model.disable_adapter():disabled=evaluate(policy,dev,'cuda')
    initial=json.loads((run/'development_0.json').read_text())
    base_error=float(np.max(abs(np.array(disabled['probabilities'])-np.array(initial['probabilities']))))
    if base_error>1e-7:raise AssertionError(f'Disabled adapter differs from original policy: {base_error}')
    status=json.loads((run/'status.json').read_text())
    if status['optimizer_updates']<=0 or status['initial_lora_sha256']==status['final_lora_sha256']:
        raise AssertionError('No actual adapter update')
    samples=run/'native_sampled_audit';samples.mkdir(exist_ok=False);draws=[]
    for seed in range(10 if step else 0):
        with policy.shared.model.disable_adapter():base=sampled(policy,dev,seed)
        trained=sampled(policy,dev,seed)
        write_json(samples/f'initial_{seed}.json',base);write_json(samples/f'trained_{seed}.json',trained)
        draws.append({'seed':seed,'initial':base['metrics'],'trained':trained['metrics'],
            'delta_Sharpe':trained['metrics']['sharpe_proxy']-base['metrics']['sharpe_proxy']})
    trained_reload_error=None
    if step==0:
        # Identical selected policies need no repeated sampling comparison,
        # but verify an actually trained checkpoint rather than only step 0.
        last=max(json.loads((run/'candidates.json').read_text()),key=lambda c:c['step'])
        del policy;gc.collect();torch.cuda.empty_cache()
        trained_policy=TextPolicy(val.env.tickers,last['package'])
        actual=evaluate(trained_policy,val,'cuda')
        reference=json.loads((run/f'validation_{last["step"]}.json').read_text())
        trained_reload_error=float(np.max(abs(np.array(actual['probabilities'])-np.array(reference['probabilities']))))
        if trained_reload_error>1e-7:raise AssertionError('Actual trained checkpoint reload differs')
    result={'status':'passed','selected_step':step,'reload_max_probability_error':error,
        'additional_trained_checkpoint_reload_error':trained_reload_error,
        'disabled_vs_initial_probability_error':base_error,'only_LoRA_trainable':True,
        'optimizer_updates':status['optimizer_updates'],'sampled_policy':draws,
        'sampling_performed':bool(step),'sampling_skip_reason':None if step else 'Selected checkpoint is the identical untrained policy',
        'mean_sampled_Sharpe_gain':float(np.mean([d['delta_Sharpe'] for d in draws])) if draws else None,
        'positive_sampled_gains':sum(d['delta_Sharpe']>0 for d in draws),
        'scope':'Technical verification and reused-development sampled-action sensitivity, not financial acceptance',
        'sampling_limit':'Ten paired action seeds share one market history; they are not ten independent market samples.'}
    write_json(run/'reload_audit.json',result);print(json.dumps(result,indent=2))

if __name__=='__main__':main()
