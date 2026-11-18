"""Final experiment: sparse, costed stock rotations around the buy-and-hold portfolio."""
import argparse, copy, datetime, hashlib, json, os, time, traceback
from pathlib import Path
import numpy as np
import torch
from research.counterfactual_rl import market
from research.counterfactual_rl.policy import NativePolicy
from research.counterfactual_rl.train import BASE, load, optimizer, write, bootstrap

EQUAL = np.full(49, 1 / 49)
YEARS = (2023, 2024)
SEEDS = (42, 123, 456)
STEPS = 768
INSTRUCTION = ('For every stock, rate its attractiveness relative to the present holdings, '
               'A strongest sell, B sell, C slight sell, D hold, E slight buy, F buy, G strongest buy. '
               'One funded seller-to-buyer rotation of up to 2 percent of equity is allowed per 20 sessions. '
               'Keep all holdings when evidence is weak. Next opening prices are unknown.')

class RotationMarket(market.Market):
    @staticmethod
    def execute_order(cash, qty, price, order, receivable=0.):
        if not isinstance(order, tuple):
            return market.execute(cash, qty, price, order, receivable)
        seller, buyer = order
        value = cash + qty @ price + receivable
        cost = price[buyer] * (1 + market.SLIPPAGE) * (1 + market.FEE)
        limit = .02 * value
        sold = min(int(qty[seller]), int(limit / price[seller]))
        proceeds = sold * price[seller] * (1 - market.SLIPPAGE) * (1 - market.FEE)
        capacity = max(0., .1 * value * .996 - qty[buyer] * price[buyer])
        bought = int(min(capacity / price[buyer], limit / cost, (cash + proceeds) / cost))
        trades = np.zeros(49, dtype=int)
        if seller == buyer or not sold or not bought:
            return cash, qty.copy(), 0., 0., trades
        trades[seller], trades[buyer] = -sold, bought
        fees = market.FEE * (sold * price[seller] * (1-market.SLIPPAGE) + bought * price[buyer] * (1+market.SLIPPAGE))
        slip = market.SLIPPAGE * (sold * price[seller] + bought * price[buyer])
        end_cash = cash + proceeds - bought * cost
        assert end_cash >= -1e-7 and (qty+trades >= 0).all()
        assert np.isclose(end_cash + (qty+trades)@price + fees + slip, value-receivable)
        return end_cash, qty + trades, fees, slip, trades

def distribution(policy, obs):
    probs = policy(policy.encode(obs))
    rating = (probs * torch.arange(-3,4,device=probs.device)).sum(1)
    # Compare each buy/sell pair on the same ordinal scale. No-trade has its own
    # fixed mass equal to the aggregate pair mass when there is no preference.
    rows = obs['stocks']
    budget = obs['equity'] * .02
    sell = np.array([r['held_shares']>0 and r['historical_quote']<=budget for r in rows])
    buy = np.array([r['historical_quote']*1.0015005<=budget and r['position_value']+r['historical_quote']<=.0996*obs['equity'] for r in rows])
    i,j = np.where(sell[:,None] & buy[None,:] & ~np.eye(49,dtype=bool))
    if not len(i): return rating[:1]*0 + 1, [None]
    utilities = (rating[j]-rating[i])*2
    # Assign half the sampling mass to no trade at equal ratings. Greedy
    # evaluation compares economic utilities, with a fixed 0.5-rating deadband.
    logits = torch.cat([utilities.new_tensor([1.+np.log(len(i))]), utilities])
    return logits.softmax(0), [None]+list(zip(i.tolist(),j.tolist()))

def greedy(p,orders):
    if len(orders)==1:return None
    i=int(p[1:].argmax())+1
    return orders[i] if p[i] > p[0]/(len(orders)-1) else None

def simulate(m,policy,start,end,capital):
    t,last=m.bounds(start,end); cash=float(capital);qty=np.zeros(49,int);pending=np.zeros(2);path=[];records=[];fees=slip=0.
    while t<last:
        obs=m.observe(t,cash,qty,capital,pending)
        if not path: order=EQUAL
        elif policy is None: order=None
        else:
            with torch.no_grad(): order=greedy(*distribution(policy,obs))
        r=m.horizon(t,cash,qty,order,min(last,t+20),pending)
        records.append(dict(date=obs['asof'],order='initial_equal' if isinstance(order,np.ndarray) else order,trades=r['trade_shares'].tolist(),cash=cash))
        cash,qty,pending=r['cash'],r['qty'],r['pending'];path.extend(r['values'].tolist());fees+=r['fees'];slip+=r['slippage'];t=min(last,t+20)
    metrics=market.metrics(path,capital,fees,slip);metrics['return']=metrics.pop('return_')
    return dict(metrics=metrics,daily_equity=path,decisions=records)

def run(args):
    root=Path(args.run);root.mkdir(exist_ok=args.resume);m=RotationMarket()
    protocol=dict(seeds=SEEDS,steps=STEPS,group=16,train='July 2018 through 2022',validation=YEARS,test='2025-01-01 to 2026-01-30 retrospective',primary_capital=100000,objective='Paired 20-session log growth vs exact no trade. Center 16 sampled action rewards with leave-one-out baseline. Reference KL 0.01, reward scale 0.001.',selection='Highest mean annual validation net log growth among RL checkpoints, earliest tie. Separately record whether both raw and holding are beaten. Report every seed regardless.',action='Identical equal-weight opening basket for all policies, followed by one feasible funded pair rotation or hold each 20 sessions. All 49 scored. Cap applies to new buys, existing holdings may drift.',inputs='Same 25 causal features and frozen InternVL3-2B backbone, fresh bridge and attention/vocabulary LoRA. No momentum imitation.',data=m.manifest['original_sha256'],code={f:hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in [__file__,'research/counterfactual_rl/market.py','research/counterfactual_rl/policy.py']})
    if args.resume:assert json.loads((root/'protocol.json').read_text())['code']==protocol['code']
    else:write(root/'protocol.json',protocol)
    first,last=m.bounds('2018-07-02','2022-12-31')
    for seed in SEEDS:
        out=root/f'seed{seed}';out.mkdir(exist_ok=True)
        if (out/'selection.json').exists():continue
        torch.manual_seed(seed);rng=np.random.default_rng(seed);p=NativePolicy(BASE);p.action_instruction=INSTRUCTION;opt=optimizer(p,'sft');params=[v for v in p.parameters() if v.requires_grad]
        for group in opt.param_groups:group['lr']*=.1
        def validate():
            return [simulate(m,p,f'{y}-01-01',f'{y}-12-31',100000)['metrics']['return'] for y in YEARS]
        candidates=[];begin=0;acct=None
        if args.resume and (out/'recovery.pt').exists():
            snap=torch.load(out/'recovery.pt',weights_only=False,map_location='cpu')
            with torch.no_grad():
                for n,v in p.named_parameters():
                    if v.requires_grad:v.copy_(snap['params'][n])
            opt.load_state_dict(snap['opt']);rng.bit_generator.state=snap['rng'];torch.cuda.set_rng_state(snap['cuda']);begin=snap['step'];acct=snap['account'];candidates=snap['candidates']
        else:
            write(out/'raw_validation.json',validate())
        for step in range(begin+1,STEPS+1):
            if acct is None:
                t=int(rng.integers(first,last-80));capital=float(rng.choice([50000,100000,200000]));r=m.horizon(t,capital,np.zeros(49,int),EQUAL,t+20);t+=20;decision=0
            else:t,capital,r,decision=acct
            cash,q,pending=r['cash'],r['qty'],r['pending'];obs=m.observe(t,cash,q,capital,pending)
            with torch.no_grad(),p.model.disable_adapter(): ref,orders=distribution(p,obs)
            prob,orders=distribution(p,obs);sample=torch.multinomial(prob.detach(),16,replacement=True);idx=sample.tolist()
            simulated={i:m.horizon(t,cash,q,orders[i],t+20,pending) for i in {0,*idx}}
            reward=np.array([np.log(simulated[i]['values'][-1]/simulated[0]['values'][-1]) for i in idx])
            adv=torch.as_tensor((reward-reward.mean())*16/15/.001,dtype=prob.dtype,device=prob.device)
            loss=-(adv*prob[sample].log()).mean()+.01*(prob*(prob.log()-ref.log())).sum()
            opt.zero_grad();loss.backward();norm=float(torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True));opt.step()
            acct=(t+20,capital,simulated[idx[0]],decision+1) if decision<2 else None
            if step%96==0:
                returns=validate();name=f'rl_{step}';candidates=[v for v in candidates if v['checkpoint']!=name];candidates.append(dict(checkpoint=name,returns=returns,score=float(np.log1p(returns).mean())))
                if not (out/name).exists():p.save(out/name,protocol)
                write(out/'candidates.json',candidates)
            if step%16==0 or step==1:
                snap=dict(params={n:v.detach().cpu() for n,v in p.named_parameters() if v.requires_grad},opt=opt.state_dict(),rng=rng.bit_generator.state,cuda=torch.cuda.get_rng_state(),step=step,account=acct,candidates=candidates)
                tmp=out/'recovery.tmp';torch.save(snap,tmp);tmp.replace(out/'recovery.pt')
                write(root/'status.json',dict(state='financial_rl',pid=os.getpid(),seed=seed,step=step,total=STEPS,grad_norm=norm,reward_std=float(reward.std()),eligible_pairs=len(orders)-1,signal=obs['asof']))
        write(out/'selection.json',max(candidates,key=lambda x:x['score']))
        del p,opt,params
    for seed in SEEDS:
        out=root/f'seed{seed}';torch.manual_seed(seed);p=NativePolicy(BASE);p.action_instruction=INSTRUCTION
        selection=json.loads((out/'selection.json').read_text());load(p,out/selection['checkpoint'])
        replay=[simulate(m,p,f'{y}-01-01',f'{y}-12-31',100000)['metrics']['return'] for y in YEARS];assert np.allclose(replay,selection['returns'],rtol=0,atol=1e-10)
        results=[]
        for capital in [500,5000,50000,100000,200000]:
            paths={}
            for name in ['rl','untrained','holding','equal_rebalanced','momentum']:
                if name=='untrained':
                    with p.model.disable_adapter():r=simulate(m,p,'2025-01-01','2026-01-30',capital)
                elif name in ['rl','holding']:r=simulate(m,p if name=='rl' else None,'2025-01-01','2026-01-30',capital)
                else:r=market.evaluate(m,market.baseline(name),'2025-01-01','2026-01-30',capital)
                write(out/f'test_{capital}_{name}.json',r);results.append(dict(capital=capital,policy=name,**r['metrics']));paths[name]=r['daily_equity']
            if capital==100000:write(out/'uncertainty.json',{k:bootstrap(paths['rl'],paths[k],capital) for k in ['holding','untrained']})
        write(out/'test_metrics.json',results)
        del p
    write(root/'status.json',dict(state='completed',pid=os.getpid()))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);ap.add_argument('--resume',action='store_true');args=ap.parse_args();torch.set_num_threads(2)
    try:run(args)
    except BaseException as exc:
        write(Path(args.run)/'status.json',dict(state='failed',error=repr(exc),traceback=traceback.format_exc()));raise
