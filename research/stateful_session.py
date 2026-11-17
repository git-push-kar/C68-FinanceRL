"""Incremental research simulator; no deployed checkpoint or cached actions.

Inject the exact selected Market and native policy.choose after result review.
This module intentionally does not load a GPU model or promote a candidate.
"""
import copy
import numpy as np

class StatefulSession:
 def __init__(self,market,choose,start,end,capital,horizon=20):
  if not np.isfinite(capital) or capital<=0:raise ValueError('Positive finite capital required')
  if horizon<=0:raise ValueError('Positive horizon required')
  self.market=market;self.choose=choose;self.initial=float(capital);self.cash=float(capital);self.qty=np.zeros(len(market.tickers),np.int64)
  self.index,self.last=market.bounds(start,end);self.horizon=int(horizon);self.sessions=0;self.values=[];self.decisions=[];self.fees=0.;self.slippage=0.;self.model_calls=0

 def advance(self):
  if self.index>=self.last:return False
  from importlib import import_module
  # The market instance's own module is authoritative for its action decoder.
  decode=import_module(type(self.market).__module__).targets
  t=self.index;decision_day=self.sessions%self.horizon==0;weights=keep=None;record=None
  if decision_day:
   observation=self.market.observe(t,self.cash,self.qty,self.initial)
   actions,receipt=self.choose(copy.deepcopy(observation))
   actions=np.asarray(actions)
   if actions.shape!=self.qty.shape or not np.issubdtype(actions.dtype,np.integer) or (actions<0).any() or (actions>=7).any():raise ValueError('Invalid model action vector; no fallback')
   weights=decode(actions,self.qty,self.cash,self.market.close[t]);keep=actions==0
   record={'signal_date':str(self.market.dates[t]),'execution_date':str(self.market.dates[t+1]),'observation':observation,'actions':actions.tolist(),'target_weights':weights[0].tolist(),'receipt':copy.deepcopy(receipt)}
  result=self.market.horizon(t,self.cash,self.qty,weights,t+1,keep=keep)
  cash=float(result['cash'][0]);qty=result['qty'][0].copy();value=float(result['values'][0,0]);fees=float(result['fees'][0]);slip=float(result['slippage'][0])
  if cash<0 or (qty<0).any() or not np.isfinite(value):raise AssertionError('Invalid execution state')
  if record is not None:
   record.update(trade_shares=result['trade_shares'][0].tolist(),cash_after_execution=cash,shares_after_execution=qty.tolist(),fees=fees,slippage=slip)
   self.decisions.append(record);self.model_calls+=1
  self.cash=cash;self.qty=qty;self.index=t+1;self.sessions+=1;self.values.append(value);self.fees+=fees;self.slippage+=slip
  return True

 def snapshot(self):
  observation=self.market.observe(self.index,self.cash,self.qty,self.initial)
  return {'asof':str(self.market.dates[self.index]),'sessions':self.sessions,'completed':self.index>=self.last,'initial':self.initial,'cash':self.cash,'equity':observation['equity'],'net_return':observation['equity']/self.initial-1,'holdings':copy.deepcopy(observation['stocks']),'fees':self.fees,'slippage':self.slippage,'model_calls':self.model_calls,'daily_equity':self.values.copy(),'decisions':copy.deepcopy(self.decisions)}
