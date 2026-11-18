"""Whole-share, next-open execution and causal observations, independent of ML."""
from pathlib import Path
import itertools,json,hashlib
import numpy as np

ROOT=Path(__file__).resolve().parent
LABELS='ABCDEFG'
# A preserves that stock's current weight; B-G request the stated equity weights.
LEVELS=np.array([np.nan,0.,.05,.10,.20,.30,1.])
ACTIONS=np.array(list(itertools.product(range(7),repeat=5)),dtype=np.int16)
FEE=.001
SLIPPAGE=.0005
HORIZON=20
TRAIN_END='2023-12-31'
SECTORS=[[0],[1,3],[2,4]]

def concentration_limits(equity,price):
 equity=np.atleast_1d(equity);affordable=equity[:,None]>=price[None,:]*(1+SLIPPAGE)*(1+FEE)
 assets=np.maximum(affordable.sum(1),1);groups=np.maximum(sum(affordable[:,g].any(1) for g in SECTORS),1)
 return np.maximum(.35,1/assets),np.maximum(.60,1/groups)

NAMES=['Coal India (coal)','HDFC Bank (bank)','Infosys (IT)','State Bank of India (bank)','Tata Consultancy Services (IT)']

def execute(cash,qty,price,target,keep=None,enforce_risk=True):
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
 if enforce_risk:
  stock_cap,sector_cap=concentration_limits(equity,price)
  conservative=equity*(1-2*(SLIPPAGE+FEE*(1+SLIPPAGE)))
  stock_budget=np.where(stock_cap<1,conservative,equity)
  desired=np.minimum(desired,np.floor(stock_cap[:,None]*stock_budget[:,None]/price+1e-10).astype(np.int64))
  for group in SECTORS:
   value=(desired[:,group]*price[group]).sum(1);budget=sector_cap*np.where(sector_cap<1,conservative,equity)
   factor=np.minimum(1.,budget/np.maximum(value,1e-30));desired[:,group]=np.floor(desired[:,group]*factor[:,None]+1e-10).astype(np.int64)
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
 stock_cap,_=concentration_limits(np.array([equity]),np.asarray(close))
 weights=np.where(actions==6,stock_cap[0],weights)
 return weights/np.maximum(weights.sum(1,keepdims=True),1.)

class Market:
 def __init__(self,directory=None):
  directory=Path(directory or ROOT.parent/'stateful_rl_20261117/data_v2');self.manifest=json.loads((directory/'manifest.json').read_text())
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
    'return60':float(self.index[t,i]/self.index[t-60,i]-1),'return120':float(self.index[t,i]/self.index[t-120,i]-1),'drawdown60':float(1-self.index[t,i]/self.index[t-59:t+1,i].max()),'vol20':float(recent.std()*np.sqrt(252)),
    'volume_ratio':float(self.volume[t,i]/max(mean_volume,1.)),
    'affordable_shares_at_signal':int(cash//(close[i]*(1+SLIPPAGE)*(1+FEE)))})
  caps=concentration_limits(np.array([equity]),close)
  return {'max_stock_weight':float(caps[0][0]),'max_sector_weight':float(caps[1][0]),'asof':str(self.dates[t]),'cash':float(cash),'equity':equity,'initial':float(initial),'stocks':rows}

 def horizon(self,t,cash,qty,weights,end,keep=None,enforce_risk=True):
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
    if weights is not None:c,q,fees,slips=execute(c,q,self.open[day],weights,keep=keep,enforce_risk=enforce_risk)
    bought=q-before
   values.append(c+(q*self.close[day]).sum(1))
  if not values:raise ValueError('No future reward sessions')
  values=np.array(values)
  return {'cash':c,'qty':q,'values':values,'reward':np.log(values[-1]/start),'fees':fees,'slippage':slips,'trade_shares':bought}

 def sample_states(self,n,seed):
  rng=np.random.default_rng(seed);first,last=self.bounds('2015-01-01',TRAIN_END);states=[]
  # Random valid inventories diversify cash, holdings and budget; they do not use
  # future returns. This is an offline, one-decision-horizon RL state distribution.
  for k in range(n):
   t=int(rng.integers(first,last-HORIZON+1));initial=float(rng.choice([500,5000,50000,100000,200000]))
   w=rng.dirichlet(np.ones(6))[:5] if k%3 else np.zeros(5)
   c,q,_,_=execute(initial,np.zeros(5,np.int64),self.close[t],w)
   cash=float(c[0]);qty=q[0];obs=self.observe(t,cash,qty,initial)
   w=targets(ACTIONS,qty,cash,self.close[t]);result=self.horizon(t,cash,qty,w,t+HORIZON,keep=ACTIONS==0);path=np.vstack([np.full(len(ACTIONS),cash+qty@self.close[t]),result['values']]);dd=(1-path/np.maximum.accumulate(path,axis=0)).max(0);reward=result['reward']-.1*dd
   states.append({'t':t,'cash':cash,'qty':qty.tolist(),'initial':initial,'observation':obs,'reward':reward.astype(np.float32)})
  return states

def prompt(observation):
 o=observation
 lines=['Manage this Indian stock portfolio using only the supplied information. Currency INR. No leverage or shorts. Whole shares only.',
  'Choose an action for EACH stock. A=keep its current share quantity; B=sell all; C=target 5%; D=10%; E=20%; F=30%; G=the maximum permitted stock weight stated below.',
  'If total requested weights exceed 100%, they are proportionally normalized. Unallocated funds remain cash. Unaffordable fractional shares cannot be bought.',
  'Orders execute at the NEXT session open, whose price is unknown now. Sell first, then fund buys from available cash. Fee 0.10% plus slippage 0.05% each way.',
  'Reconsider after 20 sessions. Grow net wealth while controlling drawdowns. Seek net account growth after costs. Historical indicators are not future returns.',
  f"Concentration limits: each stock {100*o['max_stock_weight']:.0f}% maximum; each sector {100*o['max_sector_weight']:.0f}% maximum. Sectors: coal; banks (HDFC,SBI); IT (Infosys,TCS). Limits override KEEP if necessary.\nAs of {o['asof']}. Initial capital {o['initial']:.2f}; current equity {o['equity']:.2f}; available cash {o['cash']:.2f}.",
  'Stock | business | closing price | held shares | weight | return5 | return20 | return60 | return120 | drawdown60 | annual vol20 | volume/20d mean | shares affordable with cash now']
 for s in o['stocks']:
  lines.append(f"{s['ticker']} | {s['company']} | {s['price']:.2f} | {s['shares']} | {100*s['weight']:.1f}% | {100*s['return5']:.1f}% | {100*s['return20']:.1f}% | {100*s['return60']:.1f}% | {100*s['return120']:.1f}% | {100*s['drawdown60']:.1f}% | {100*s['vol20']:.1f}% | {s['volume_ratio']:.2f} | {s['affordable_shares_at_signal']}")
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
  result=market.horizon(t,c,q,w,min(t+HORIZON,last),keep=keep,enforce_risk=not getattr(choose,'single_stock_hold',False))
  c=float(result['cash'][0]);q=result['qty'][0];fees+=float(result['fees'][0]);slips+=float(result['slippage'][0]);values.extend(result['values'][:,0].tolist())
  records.append({'signal_date':str(market.dates[t]),'execution_date':str(market.dates[t+1]),'observation':obs,'actions':None if actions is None else np.asarray(actions).tolist(),'target_weights':None if w is None else w[0].tolist(),'trade_shares':result['trade_shares'][0].tolist(),'cash_after_horizon':c,'shares_after_horizon':q.tolist(),'receipt':receipt})
  t=min(t+HORIZON,last)
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
 choose.single_stock_hold=kind.startswith('single_hold_')
 return choose
