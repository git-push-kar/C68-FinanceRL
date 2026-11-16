"""Fresh CPU reload and sampling check for a completed numerical control."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from numeric_policy import NumericPolicy
from candidate_market import Market,evaluate,tensor
from metrics import metrics_from_infos
from model import write_json
from execution_policy import wait_for_compute_window

@torch.no_grad()
def sampled(policy,market,seed,initial=False):
    rng=np.random.default_rng(seed);state=market.reset()
    while True:
        wait_for_compute_window();obs,targets=state
        p=np.ones(4)/4 if initial else policy(tensor(obs,'cpu'))[0].probs[0].numpy().astype(float)
        p/=p.sum();action=min(int(np.searchsorted(np.cumsum(p),rng.random())),3)
        state,_,done=market.step(targets[action])
        if done:break
    return {'metrics':metrics_from_infos(market.infos),'daily':market.infos}

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);a=p.parse_args();run=Path(a.run)
    torch.set_num_threads(2);status=json.loads((run/'status.json').read_text())
    if status['status']!='completed' or status['initial_policy_sha256']==status['final_policy_sha256']:
        raise AssertionError('No completed learning run')
    selected=json.loads((run/'selection.json').read_text())['selected'];step=selected['step']
    data='../data/nifty50_prices.csv'
    val=Market(data,'2022-01-01','2022-12-31');dev=Market(data,'2023-01-01','2024-12-31')
    policy=NumericPolicy(val.env.tickers,selected['package']).eval()
    observed=evaluate(policy,val,'cpu');expected=json.loads((run/f'validation_{step}.json').read_text())
    error=float(np.max(abs(np.array(observed['probabilities'])-np.array(expected['probabilities']))))
    if not np.isfinite(error) or error>1e-7:raise AssertionError('Reload differs')
    draws=[];out=run/'numeric_sampled_audit';out.mkdir(exist_ok=False)
    for seed in range(10 if step else 0):
        initial=sampled(policy,dev,seed,True);trained=sampled(policy,dev,seed)
        write_json(out/f'initial_{seed}.json',initial);write_json(out/f'trained_{seed}.json',trained)
        draws.append({'seed':seed,'delta_Sharpe':trained['metrics']['sharpe_proxy']-initial['metrics']['sharpe_proxy']})
    extra=None
    if step==0:
        last=max(json.loads((run/'candidates.json').read_text()),key=lambda c:c['step'])
        trained=NumericPolicy(val.env.tickers,last['package']).eval()
        actual=evaluate(trained,val,'cpu');reference=json.loads((run/f'validation_{last["step"]}.json').read_text())
        extra=float(np.max(abs(np.array(actual['probabilities'])-np.array(reference['probabilities']))))
        if not np.isfinite(extra) or extra>1e-7:raise AssertionError('Actual trained reload differs')
    result={'status':'passed','model':'numerical prompt-summary MLP, not an adapter',
        'selected_step':step,'reload_max_probability_error':error,'additional_trained_reload_error':extra,
        'sampled_gains':draws,'mean_sampled_gain':float(np.mean([d['delta_Sharpe'] for d in draws])) if draws else None,
        'scope':'Development only; action draws are not independent market histories'}
    write_json(run/'reload_audit.json',result);print(json.dumps(result,indent=2))

if __name__=='__main__':main()
