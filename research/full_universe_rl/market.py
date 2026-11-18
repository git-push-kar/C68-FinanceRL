"""All original49 assets; original signals, reconstructed whole-share execution.

Approximations are explicit in execution_audit.json. No original data overwritten.
"""
import hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
from state import OriginalPanel,ROOT

LEVELS=np.array([0.,.005,.01,.02,.04,.07,.10])
FEE=.001
SLIPPAGE=.0005
HORIZON=20
EVENTS=[dict(parent='RELIANCE.NS',child='JIOFIN.NS',ex='2023-07-20',prior='2023-07-19',prior_close=2841.85,ratio=1.,mark=261.85,listing='2023-08-21'),
 dict(parent='ITC.NS',child='ITCHOTELS.NS',ex='2025-01-06',prior='2025-01-03',prior_close=481.60,ratio=.1,mark=260.,listing='2025-01-29')]

def execute(cash,qty,price,weights,receivable=0.):
 q=np.asarray(qty,np.int64).copy();c=float(cash);price=np.asarray(price,float);w=np.asarray(weights,float).copy()
 if q.shape!=(49,) or w.shape!=(49,) or not np.isfinite(w).all() or (w<0).any():raise ValueError('Explicit49 nonnegative weights required')
 if not np.isfinite(price).all() or (price<=0).any():raise ValueError('Bad quotes')
 w=np.minimum(w,.10);w/=max(w.sum(),1.)
 equity=c+q@price+receivable
 desired=np.floor(w*equity/price+1e-10).astype(np.int64)
 sell=np.maximum(q-desired,0);q-=sell;c+=float(sell@price)*(1-SLIPPAGE)*(1-FEE)
 buy=np.maximum(desired-q,0);unit=price*(1+SLIPPAGE)*(1+FEE)
 buy=np.floor(buy*min(1.,c/max(float(buy@unit),1e-30))+1e-10).astype(np.int64)
 c-=float(buy@unit);q+=buy
 fees=float((sell@price*(1-SLIPPAGE)+buy@price*(1+SLIPPAGE))*FEE);slips=float((sell+buy)@price*SLIPPAGE)
 assert c>=-1e-7 and (q>=0).all()
 assert np.isclose(equity,c+q@price+receivable+fees+slips,atol=1e-7,rtol=1e-10)
 return max(c,0.),q,fees,slips,buy-sell

def read_capture(ticker):
 frame=pd.read_csv(ROOT/'execution_fields'/f'{ticker}.csv',index_col=0)
 frame.index=pd.to_datetime(frame.index,utc=True).tz_convert('Asia/Kolkata').tz_localize(None).normalize()
 return frame

class Market(OriginalPanel):
 def __init__(self):
  super().__init__();shape=self.prices.shape
  self.open=np.full(shape,np.nan);self.close=np.full(shape,np.nan);self.dividends=np.zeros(shape);self.splits=np.ones(shape);self.events=[];audit=[]
  for i,ticker in enumerate(self.tickers):
   f=read_capture(ticker);split=f['Stock Splits'].replace(0,1).to_numpy(float)
   # Yahoo OHLC and dividends use latest split units even with auto_adjust=False.
   future=np.cumprod(split[::-1])[::-1]/split
   op=f.Open.to_numpy()*future;cl=f.Close.to_numpy()*future;div=f.Dividends.to_numpy()*future
   factors=[]
   for ev in EVENTS:
    if ev['parent']==ticker:
     j=f.index.get_loc(pd.Timestamp(ev['prior']));factor=ev['prior_close']/cl[j]
     before=f.index<pd.Timestamp(ev['ex']);op[before]*=factor;cl[before]*=factor
     # Demerger adjustments are not share-count changes. Do not multiply dividends.
     factors.append({'ex':ev['ex'],'factor':factor,'verified_prior_close':ev['prior_close']})
   lookup=f.index.get_indexer(pd.to_datetime(self.dates));valid=lookup>=0;ix=lookup[valid]
   self.open[valid,i]=op[ix];self.close[valid,i]=cl[ix];self.dividends[valid,i]=div[ix];self.splits[valid,i]=split[ix]
   audit.append({'ticker':ticker,'capture_sha256':hashlib.sha256((ROOT/'execution_fields'/f'{ticker}.csv').read_bytes()).hexdigest(),'demerger_reconstruction':factors})
  for ev in EVENTS:
   f=read_capture(ev['child']);e=dict(ev);e['parent_index']=self.tickers.index(ev['parent']);e['listing_open']=float(f.loc[pd.Timestamp(ev['listing']),'Open']);self.events.append(e)
  recent=self.dates>=np.datetime64('2018-07-02')
  assert np.isfinite(self.open[recent]).all() and np.isfinite(self.close[recent]).all()
  assert (self.open[recent]>0).all() and (self.close[recent]>0).all()
  self.audit={'source_sha256':self.manifest['original_sha256'],'assets':49,'events':self.events,'captures':audit,
   'assumptions':['Split/bonus ratios reconstruct historical share units; fractional bonus entitlement settled at ex-date open as explicit cash-in-lieu approximation.',
   'Demerged securities marked at published discovery value until listing, unavailable to fund buys; mechanically liquidated at listing open with identical costs for every policy. Fractional child entitlement uses same cash-in-lieu approximation.',
   'Dividends credited ex-date, not payment date; no taxes, settlement delays, liquidity limits or market impact.',
   '10percent maximum requested stock weight, including low capital. Small accounts may rationally remain cash; no fictitious fractional purchases.',
   'Original adjusted prices used only for signals; vendor OHLC reconstructed for executions. Not an exchange-certified complete price database.',
   'Trading starts July2018 to give every original ticker126 real historical observations. Earlier original rows remain preserved. Fixed contemporary universe implies survivorship bias.'],
   'sources':['https://www.ril.com/ar2022-23/pdf/RIL-Integrated-Annual-Report-2022-23.pdf','https://nsearchives.nseindia.com/corporate/ITC_30122024153104_SELetter.pdf','https://nsearchives.nseindia.com/web/sites/default/files/2023-08/PR_cc_17072023.pdf','https://www.etnownews.com/markets/itc-hotels-share-price-nse-bse-finalised-after-demerger-check-here-article-116980834']}

 def bounds(self,start,end):
  first=int(np.searchsorted(self.dates,np.datetime64(start)))-1;last=int(np.searchsorted(self.dates,np.datetime64(end),side='right'))-1
  if first<126 or last<=first or not np.isfinite(self.prices[first-126:last+1]).all():raise ValueError('Insufficient actual all49 history')
  return first,last

 def observe(self,t,cash,qty,initial,pending=None):
  pending=np.zeros(2) if pending is None else np.asarray(pending,float)
  value=float(pending@np.array([e['mark'] for e in self.events]))
  obs=super().observe(t,cash,qty,initial,self.close[t],value)
  obs['pending_entitlements']=[{'ticker':e['child'],'quantity':float(v),'marked_value':float(v*e['mark'])} for e,v in zip(self.events,pending) if v]
  obs['receivable_value']=value
  return obs

 def horizon(self,t,cash,qty,weights,end,pending=None):
  c=float(cash);q=np.asarray(qty,np.int64).copy();p=np.zeros(2) if pending is None else np.asarray(pending,float).copy()
  start=self.observe(t,c,q,1.,p)['equity'];values=[];fees=slips=0.;trade=np.zeros(49,int);ledger=[]
  for day in range(t+1,end+1):
   date=str(self.dates[day]);entitled=q*self.splits[day];whole=np.floor(entitled+1e-8).astype(np.int64);fraction=np.maximum(entitled-whole,0)
   credit=float(fraction@self.open[day]);c+=credit;q=whole
   dividend=float(q@self.dividends[day]);c+=dividend
   if credit or dividend:ledger.append({'date':date,'fractional_bonus_cash':credit,'dividend':dividend})
   for j,e in enumerate(self.events):
    if date==e['ex']:
     p[j]+=q[e['parent_index']]*e['ratio'];ledger.append({'date':date,'entitlement':e['child'],'quantity':float(p[j]),'cash_credit':0.})
    if date==e['listing'] and p[j]:
     gross=p[j]*e['listing_open'];f=gross*(1-SLIPPAGE)*FEE;s=gross*SLIPPAGE;c+=gross-f-s;fees+=f;slips+=s
     ledger.append({'date':date,'liquidation':e['child'],'quantity':float(p[j]),'gross':gross,'fees':f,'slippage':s});p[j]=0.
   if day==t+1 and weights is not None:
    receivable=float(p@np.array([e['mark'] for e in self.events]))
    c,q,f,s,trade=execute(c,q,self.open[day],weights,receivable);fees+=f;slips+=s
   values.append(float(c+q@self.close[day]+p@np.array([e['mark'] for e in self.events])))
  if not values:raise ValueError('Empty horizon')
  return {'cash':c,'qty':q,'pending':p,'values':np.array(values),'reward':float(np.log(values[-1]/start)),'fees':fees,'slippage':slips,'trade_shares':trade,'ledger':ledger}

def metrics(values,initial,fees,slips):
 v=np.r_[initial,values];r=v[1:]/v[:-1]-1
 return dict(return_=float(v[-1]/initial-1),end_equity=float(v[-1]),sharpe=float(np.sqrt(252)*r.mean()/max(r.std(),1e-12)),max_drawdown=float((1-v/np.maximum.accumulate(v)).max()),fees=float(fees),slippage=float(slips),sessions=len(r))

def evaluate(m,choose,start,end,capital):
 t,last=m.bounds(start,end);cash=float(capital);qty=np.zeros(49,np.int64);pending=np.zeros(2);values=[];records=[];fees=slips=0.
 while t<last:
  obs=m.observe(t,cash,qty,capital,pending);choice=choose(obs);actions=None;receipt={}
  if isinstance(choice,tuple):actions,receipt=choice;weights=LEVELS[np.asarray(actions,int)]
  else:weights=choice
  result=m.horizon(t,cash,qty,weights,min(t+HORIZON,last),pending)
  records.append({'signal_date':str(m.dates[t]),'execution_date':str(m.dates[t+1]),'observation':obs,'actions':None if actions is None else np.asarray(actions).tolist(),'receipt':receipt,'trade_shares':result['trade_shares'].tolist(),'ledger':result['ledger']})
  cash,qty,pending=result['cash'],result['qty'],result['pending'];values.extend(result['values'].tolist());fees+=result['fees'];slips+=result['slippage'];t=min(t+HORIZON,last)
 result=metrics(values,capital,fees,slips);result['return']=result.pop('return_')
 return {'metrics':result,'daily_equity':values,'decisions':records}

def baseline(kind):
 first=True
 def choose(obs):
  nonlocal first
  if kind=='buy_hold':
   w=np.ones(49)/49 if first else None;first=False;return w
  if kind=='equal_rebalanced':return np.ones(49)/49
  if kind=='cash':return np.zeros(49)
  if kind=='inverse_vol':
   w=1/np.maximum([r['vol20'] for r in obs['stocks']],.05);return w/w.sum()
  if kind=='momentum':
   score=np.array([r['return120'] for r in obs['stocks']]);w=np.zeros(49);w[np.argsort(score)[-10:]]=.1;return w
  raise ValueError(kind)
 return choose

if __name__=='__main__':
 m=Market();(ROOT/'execution_audit.json').write_text(json.dumps(m.audit,indent=2));print(json.dumps({'assets':49,'events':m.events,'bounds':m.bounds('2018-07-02','2023-12-31')}))
