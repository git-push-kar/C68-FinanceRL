"""Direct finance RL through the actually executed expected-exposure portfolio."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from candidate_market import Market,evaluate,tensor
from numeric_policy import NumericPolicy,state_digest
from candidate_path import path_returns
from direct_ledger import negative_sharpe
from train import seed_everything
from model import write_json,digest
from execution_policy import configure_finance_precision,wait_for_compute_window

def normalized_probabilities(policy,obs):
    p=policy(tensor(obs,'cpu'))[0].probs[0].double()
    return p/p.sum()

def collect(policy,market):
    state=market.reset()
    sessions=len(market.env.prices)-market.env.start_index-1
    observations=[];old=[];candidates=[]
    with torch.no_grad():
        e=market.env;initial_weights=e.weights.copy();initial_cash=e.cash_weight;begin=e.index;info_begin=len(market.infos)
        for offset in range(0,sessions,5):
            wait_for_compute_window();obs,_=state;p=normalized_probabilities(policy,obs)
            targets=obs['candidate_targets'];observations.append(obs);old.append(p.numpy());candidates.append(targets)
            state,_,done=market.step(p.numpy()@targets,max_days=min(5,sessions-offset))
            if done!=(offset+5>=sessions):raise RuntimeError('Full training path boundary mismatch')
    relatives=e.prices[begin+1:begin+sessions+1]/e.prices[begin:begin+sessions]
    leaves=torch.tensor(np.asarray(old),dtype=torch.float64,requires_grad=True)
    returns=path_returns(leaves,np.asarray(candidates),initial_weights,initial_cash,relatives)
    reference=np.array([x['net_return'] for x in market.infos[info_begin:]])
    error=float(np.max(abs(returns.detach().numpy()-reference)))
    if error>1e-11:raise AssertionError(f'Financial reconstruction mismatch {error}')
    return observations,old,leaves,returns,error

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--updates',type=int,default=64);parser.add_argument('--eval-every',type=int,default=4)
    a=parser.parse_args()
    if min(a.updates,a.eval_every)<1:raise ValueError('Positive budgets required')
    if torch.cuda.is_available():raise RuntimeError('CPU control requires hidden CUDA')
    seed_everything(a.seed);configure_finance_precision();torch.set_num_threads(2)
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    data='../multiasset_data_20261115/five_asset_development_through2024.csv'
    train=Market(data,'2010-01-01','2018-12-31');val=Market(data,'2019-01-01','2021-12-31');dev=Market(data,'2022-01-01','2024-12-31')
    policy=NumericPolicy(train.env.tickers);optimizer=torch.optim.AdamW(policy.parameters(),lr=.001,weight_decay=0)
    initial=state_digest(policy);candidates=[]
    metadata={'algorithm':'full_history_direct_portfolio_reward_two_pass_chain_rule','arguments':vars(a),
        'execution':{'device':'cpu','cuda_hidden':True,'threads':2},
        'training':'2010-2018','selection':'2019-2021 net daily Sharpe, initial eligible, earliest exact tie',
        'development':'2022-2024 reused exploratory','reward':'Annualized Sharpe of the complete 2010-2018 training portfolio net daily returns',
        'burn_in':'None; full path from common cash initialization',
        'regularization':'.01 mean KL to initial uniform policy, on current observed states',
        'optimization':'AdamW .001, one accumulated update per full chronological training path, norm cap 1, post-update mean KL ceiling .03',
        'actions':'Deterministic mixture of equal weight, inverse volatility, momentum, reversal, causal minimum variance; same execution in collection and evaluation',
        'limitations':'Local piecewise-smooth gradient of the finite historical path; no derivatives across rounded-state discontinuities; no population generalization or global optimality guarantee',
        'prior_art':'https://people.idsia.ch/~juergen/rnnaissance2003talks/MoodySaffellTNN01.pdf',
        'dataset_sha256':digest(data),'source_sha256':{f.name:digest(f) for f in Path(__file__).parent.glob('*.py')},
        'initial_policy_sha256':initial,'test2026_accessed':False}
    write_json(out/'protocol.json',metadata)
    def assess(step):
        v=evaluate(policy,val,'cpu');d=evaluate(policy,dev,'cpu')
        write_json(out/f'validation_{step}.json',v);write_json(out/f'development_{step}.json',d)
        package=out/f'checkpoint_{step}';policy.save(package,metadata)
        torch.save({'optimizer':optimizer.state_dict(),'update':step,'torch_rng':torch.get_rng_state()},package/'training_state.pt')
        candidates.append({'step':step,'validation':v['metrics'],'development':d['metrics'],'package':str(package.resolve())})
        write_json(out/'candidates.json',candidates);print('ASSESS',json.dumps(candidates[-1]),flush=True)
    assess(0);started=time.monotonic()
    for update in range(1,a.updates+1):
        wait_for_compute_window();segments=[collect(policy,train)]
        objective=negative_sharpe(torch.cat([s[3] for s in segments]))
        coefficients=torch.autograd.grad(objective,[s[2] for s in segments])
        batch=[(obs,old,c) for segment,coefs in zip(segments,coefficients) for obs,old,c in zip(segment[0],segment[1],coefs)]
        optimizer.zero_grad();error=0.;kl_to_initial=0.
        for obs,prior,coefficient in batch:
            wait_for_compute_window();p=normalized_probabilities(policy,obs)
            error=max(error,float(np.max(abs(p.detach().numpy()-prior))))
            kl=(p*torch.log(p/.2)).sum()
            loss=(p*coefficient.detach()).sum()+.01*kl/len(batch)
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite direct objective')
            loss.backward();kl_to_initial+=float(kl.detach())/len(batch)
        if error>1e-7:raise AssertionError('Collection/recomputation probability mismatch')
        grad=float(torch.nn.utils.clip_grad_norm_(policy.parameters(),1.,error_if_nonfinite=True));optimizer.step()
        kls=[]
        with torch.no_grad():
            for obs,prior,_ in batch:
                post=normalized_probabilities(policy,obs).numpy()
                kls.append(float(np.sum(prior*(np.log(prior)-np.log(post)))))
        kl=float(np.mean(kls))
        if not np.isfinite(kl) or kl>.03:raise RuntimeError('Post-update KL guard failed')
        row={'update':update,'optimizer_updates':update,'training_Sharpe_before_update':-float(objective.detach()),
             'KL_to_initial':kl_to_initial,'mean_post_update_KL':kl,'gradient_norm':grad,
             'initial_probability_error':error,'financial_path_max_error':max(s[4] for s in segments),
             'policy_states':len(batch),'daily_training_returns':sum(len(s[3]) for s in segments),'seconds':time.monotonic()-started}
        write_json(out/'progress.json',row);print(json.dumps(row),flush=True)
        if update%a.eval_every==0 or update==a.updates:assess(update)
    selected=max(candidates,key=lambda c:(c['validation']['sharpe_proxy'],-c['step']))
    write_json(out/'selection.json',{'selected':selected,'initial':candidates[0],'criterion':metadata['selection']})
    write_json(out/'status.json',{'status':'completed','optimizer_updates':a.updates,'initial_policy_sha256':initial,
        'final_policy_sha256':state_digest(policy),'financial_goal_met':False,'test2026_accessed':False})

if __name__=='__main__':main()
