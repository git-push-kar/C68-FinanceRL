"""Whole-share, next-open execution and causal observations, independent of ML."""
from pathlib import Path
import itertools,json,hashlib
import numpy as np

ROOT=Path(__file__).resolve().parent
LABELS='ABCDEFG'
# A preserves that stock's current weight; B-G request the stated equity weights.
LEVELS=np.array([np.nan,0.,.1,.2,.4,.6,1.])
ACTIONS=np.array(list(itertools.product(range(7),repeat=5)),dtype=np.int16)
FEE=.001
SLIPPAGE=.0005
NAMES=['Coal India (coal)','HDFC Bank (bank)','Infosys (IT)','State Bank of India (bank)','Tata Consultancy Services (IT)']

def execute(cash,qty,price,target,keep=None):
 """Vectorized sell-before-buy execution. Integer lots, no shorts/borrowing.

 Price is the actual next open, never exposed to the preceding observation.
 Buy shortages are proportionally scaled and rounded; residual cash is retained,
 avoiding a hidden alphabetical first-stock purchasing advantage.
 """
 qty=np.asarray(qty,dtype=np.int64).copy();cash=np.asarray(cash,dtype=float).copy()
 if qty.ndim==1:qty=qty[None,:]
 cash=np.broadcast_to(cash,(len(qty),)).copy();price=np.asarray(price,dtype=float)
 target=np.broadcast_to(np.asarray(target,float),qty.shape).copy()
 if not np.isfinite(target).all() or (target<0).any():raise ValueError('Invalid targets')
 target/=np.maximum(target.sum(1,keepdims=True),1.)
 equity=cash+(qty*price).sum(1)
 desired=np.floor(target*equity[:,None]/price+1e-10).astype(np.int64)
 if keep is not None:desired=np.where(keep,qty,desired)
 sells=np.maximum(qty-desired,0);sell_value=sells*price*(1-SLIPPAGE)
 cash+=sell_value.sum(1)*(1-FEE);qty-=sells
 buys=np.maximum(desired-qty,0);unit=price*(1+SLIPPAGE)*(1+FEE)
 total=(buys*unit).sum(1);scale=np.minimum(1.,cash/np.maximum(total,1e-30))
 funded=np.floor(buys*scale[:,None]+1e-10).astype(np.int64)
 # No sequential "spend everything on first ticker" residual allocation.
 spent=(funded*unit).sum(1);cash-=spent;qty+=funded
 fees=(sell_value.sum(1)+(funded*price*(1+SLIPPAGE)).sum(1))*FEE
 slippage=((sells+funded)*price).sum(1)*SLIPPAGE
 if (cash < -1e-7).any() or (qty<0).any():raise AssertionError('Funding/shorting error')
 if not np.allclose(equity,cash+(qty*price).sum(1)+fees+slippage,atol=1e-7,rtol=1e-10):raise AssertionError('Execution does not conserve money')
 cash=np.maximum(cash,0.)
 return cash,qty,fees,slippage

def targets(actions,qty,cash,close):
 actions=np.asarray(actions,int)
 if actions.ndim==1:actions=actions[None,:]
 equity=float(cash+np.dot(qty,close));current=qty*close/equity
 weights=LEVELS[actions];weights=np.where(actions==0,current,weights)
 return weights/np.maximum(weights.sum(1,keepdims=True),1.)

class Market:
 def __init__(self,directory=None):
  directory=Path(directory or ROOT/'data_v2');self.manifest=json.loads((directory/'manifest.json').read_text())
  path=directory/'market.npz'
  if hashlib.sha256(path.read_bytes()).hexdigest()!=self.manifest['sha256']['market.npz']:raise ValueError('Data hash mismatch')
  with np.load(path,allow_pickle=False) as z:
   for key in ['dates','tickers','open','close','dividends','splits','volume']:setattr(self,key,z[key].copy())
  self.tickers=self.tickers.tolist()
  self.relative=np.ones_like(self.close)
  self.relative[1:]=(self.close[1:]*self.splits[1:]+self.dividends[1:]*self.splits[1:])/self.close[:-1]
  if np.abs(self.relative[1:]-1).max()>.5:raise ValueError('Unreviewed economic price discontinuity')
  self.index=np.cumprod(self.relative,axis=0)

 def bounds(self,start,end):
  first=int(np.searchsorted(self.dates,np.datetime64(start)))-1
  last=int(np.searchsorted(self.dates,np.datetime64(end),side='right'))-1
  if first<126 or last<=first:raise ValueError('Insufficient causal history')
  return first,last

 def observe(self,t,cash,qty,initial):
  if t<126:raise ValueError('Need causal history')
  close=self.close[t];equity=float(cash+qty@close);rows=[]
  for i,ticker in enumerate(self.tickers):
   recent=self.relative[t-19:t+1,i]-1
   mean_volume=self.volume[t-19:t+1,i].mean()
   rows.append({'ticker':ticker,'company':NAMES[i],'price':float(close[i]),'shares':int(qty[i]),'weight':float(qty[i]*close[i]/equity),
    'return5':float(self.index[t,i]/self.index[t-5,i]-1),'return20':float(self.index[t,i]/self.index[t-20,i]-1),
    'return60':float(self.index[t,i]/self.index[t-60,i]-1),'vol20':float(recent.std()*np.sqrt(252)),
    'volume_ratio':float(self.volume[t,i]/max(mean_volume,1.)),
    'affordable_shares_at_signal':int(cash//(close[i]*(1+SLIPPAGE)*(1+FEE)))})
  return {'asof':str(self.dates[t]),'cash':float(cash),'equity':equity,'initial':float(initial),'stocks':rows}

 def horizon(self,t,cash,qty,weights,end,keep=None):
  """Counterfactual action returns; same routine used for actual evaluation.

 Each action trades only at t+1 open and holds until end. Corporate actions
 happen before that day's open; cash dividends are credited on ex-date.
 """
  count=1 if weights is None else len(np.atleast_2d(weights))
  if weights is not None:weights=np.atleast_2d(weights)
  q=np.broadcast_to(np.asarray(qty,np.int64),(count,5)).copy();c=np.full(count,float(cash));start=float(cash+qty@self.close[t])
  values=[];fees=np.zeros(count);slips=np.zeros(count);bought=None
  for day in range(t+1,end+1):
   q=q*self.splits[day].astype(np.int64)
   c+=(q*self.dividends[day]).sum(1)
   if day==t+1:
    before=q.copy()
    if weights is not None:c,q,fees,slips=execute(c,q,self.open[day],weights,keep=keep)
    bought=q-before
   values.append(c+(q*self.close[day]).sum(1))
  if not values:raise ValueError('No future reward sessions')
  values=np.array(values)
  return {'cash':c,'qty':q,'values':values,'reward':np.log(values[-1]/start),'fees':fees,'slippage':slips,'trade_shares':bought}

 def sample_states(self,n,seed):
  rng=np.random.default_rng(seed);first,last=self.bounds('2015-01-01','2021-12-31');states=[]
  # Random valid inventories diversify cash, holdings and budget; they do not use
  # future returns. This is an offline, one-decision-horizon RL state distribution.
  for k in range(n):
   t=int(rng.integers(first,last-4));initial=float(rng.choice([500,5000,50000,100000,200000]))
   w=rng.dirichlet(np.ones(6))[:5] if k%3 else np.zeros(5)
   c,q,_,_=execute(initial,np.zeros(5,np.int64),self.close[t],w)
   cash=float(c[0]);qty=q[0];obs=self.observe(t,cash,qty,initial)
   w=targets(ACTIONS,qty,cash,self.close[t]);reward=self.horizon(t,cash,qty,w,t+5,keep=ACTIONS==0)['reward']
   states.append({'t':t,'cash':cash,'qty':qty.tolist(),'initial':initial,'observation':obs,'reward':reward.astype(np.float32)})
  return states

def prompt(observation):
 o=observation
 lines=['Manage this Indian stock portfolio using only the supplied information. Currency INR. No leverage or shorts. Whole shares only.',
  'Choose an action for EACH stock. A=keep its current share quantity; B=sell all; C=target 10%; D=20%; E=40%; F=60%; G=100% of account equity.',
  'If total requested weights exceed 100%, they are proportionally normalized. Unallocated funds remain cash. Unaffordable fractional shares cannot be bought.',
  'Orders execute at the NEXT session open, whose price is unknown now. Sell first, then fund buys from available cash. Fee 0.10% plus slippage 0.05% each way.',
  'Reconsider after five sessions. Seek net account growth after costs. Historical indicators are not future returns.',
  f"As of {o['asof']}. Initial capital {o['initial']:.2f}; current equity {o['equity']:.2f}; available cash {o['cash']:.2f}.",
  'Stock | business | closing price | held shares | weight | return5 | return20 | return60 | annual vol20 | volume/20d mean | shares affordable with cash now']
 for s in o['stocks']:
  lines.append(f"{s['ticker']} | {s['company']} | {s['price']:.2f} | {s['shares']} | {100*s['weight']:.1f}% | {100*s['return5']:.1f}% | {100*s['return20']:.1f}% | {100*s['return60']:.1f}% | {100*s['vol20']:.1f}% | {s['volume_ratio']:.2f} | {s['affordable_shares_at_signal']}")
 lines.append('Provide the action letter for every allocation slot below, using this current account and these stock signals.')
 return '\n'.join(lines)

def metrics(values,initial,fees,slippage):
 v=np.r_[float(initial),np.asarray(values,float)];r=v[1:]/v[:-1]-1
 return {'return':float(v[-1]/initial-1),'end_equity':float(v[-1]),'sharpe':float(np.sqrt(252)*r.mean()/max(r.std(),1e-12)),
  'max_drawdown':float((1-v/np.maximum.accumulate(v)).max()),'annual_volatility':float(r.std()*np.sqrt(252)),
  'fees':float(fees),'slippage':float(slippage),'sessions':len(r)}

def evaluate(market,choose,start,end,capital):
 t,last=market.bounds(start,end);c=float(capital);q=np.zeros(5,np.int64);values=[];records=[];fees=slips=0.
 while t<last:
  obs=market.observe(t,c,q,capital);decision=choose(obs)
  if isinstance(decision,tuple):actions,receipt=decision;w=targets(actions,q,c,market.close[t]);keep=np.asarray(actions)==0
  else:w=None if isinstance(decision,str) and decision=='hold' else np.atleast_2d(decision);actions=None;receipt={};keep=None
  result=market.horizon(t,c,q,w,min(t+5,last),keep=keep)
  c=float(result['cash'][0]);q=result['qty'][0];fees+=float(result['fees'][0]);slips+=float(result['slippage'][0]);values.extend(result['values'][:,0].tolist())
  records.append({'signal_date':str(market.dates[t]),'execution_date':str(market.dates[t+1]),'observation':obs,'actions':None if actions is None else np.asarray(actions).tolist(),'target_weights':None if w is None else w[0].tolist(),'trade_shares':result['trade_shares'][0].tolist(),'cash_after_horizon':c,'shares_after_horizon':q.tolist(),'receipt':receipt})
  t=min(t+5,last)
 return {'metrics':metrics(values,capital,fees,slips),'daily_equity':values,'decisions':records}

def baseline(kind):
 first=True
 def choose(obs):
  nonlocal first
  if kind.startswith('single_hold_'):
   w=np.zeros(5);w[int(kind.rsplit('_',1)[1])]=1
   result=w if first else 'hold';first=False;return result
  if kind=='buy_hold':
   w=np.ones(5)/5 if first else 'hold';first=False;return w
  if kind=='equal_rebalanced':return np.ones(5)/5
  if kind=='cash':return np.zeros(5)
  if kind=='inverse_vol':
   w=1/np.maximum([s['vol20'] for s in obs['stocks']],1e-5);return w/w.sum()
  if kind=='momentum':
   order=np.argsort([s['return60'] for s in obs['stocks']]);w=np.zeros(5);w[order[-2:]]=.5;return w
  raise ValueError(kind)
 return choose
