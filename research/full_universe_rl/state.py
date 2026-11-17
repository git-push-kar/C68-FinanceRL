"""Full original-universe observation contract, independent of training.

Adjusted price is explicitly a signal index, never labeled an executable quote.
Historical raw quotes are a separately supplied, validated input.
"""
from pathlib import Path
import hashlib,json
import numpy as np

ROOT=Path(__file__).resolve().parent

class OriginalPanel:
 def __init__(self):
  self.manifest=json.loads((ROOT/'original_dataset_audit.json').read_text())
  path=ROOT/'original_panel.npz'
  if hashlib.sha256(path.read_bytes()).hexdigest()!=self.manifest['panel_sha256']:raise ValueError('Panel integrity failure')
  with np.load(path,allow_pickle=False) as z:
   self.dates=z['dates'];self.tickers=z['tickers'].tolist();self.prices=z['adjusted_close'];self.volume=z['volume'];self.observed=z['observed']
  if self.tickers!=self.manifest['tickers'] or len(self.tickers)!=49:raise ValueError('Universe mismatch')

 def observe(self,t,cash,shares,initial,quotes):
  n=len(self.tickers);shares=np.asarray(shares);quotes=np.asarray(quotes,float)
  if t<126:raise ValueError('Insufficient causal feature history')
  if shares.shape!=(n,) or quotes.shape!=(n,):raise ValueError('Account must represent every original ticker')
  if not np.issubdtype(shares.dtype,np.integer) or (shares<0).any():raise ValueError('Whole nonnegative shares required')
  if not np.isfinite(quotes).all() or (quotes<=0).any():raise ValueError('Validated executable quote required for every ticker')
  if not np.isfinite(cash) or cash<0 or not np.isfinite(initial) or initial<=0:raise ValueError('Invalid account')
  history=self.prices[t-126:t+1]
  if not np.isfinite(history).all():raise ValueError('Do not invent pre-listing history; explicit unavailable-state handling required')
  equity=float(cash+shares@quotes);returns=history[1:]/history[:-1]-1;rows=[]
  for i,ticker in enumerate(self.tickers):
   r={'ticker':ticker,'historical_quote':float(quotes[i]),'adjusted_signal_price':float(history[-1,i]),'held_shares':int(shares[i]),'position_value':float(shares[i]*quotes[i]),'weight':float(shares[i]*quotes[i]/equity),'affordable_shares_cash':int(cash//(quotes[i]*1.0015005))}
   for horizon in [5,20,60,120]:r[f'return{horizon}']=float(history[-1,i]/history[-horizon-1,i]-1)
   r.update(vol20=float(returns[-20:,i].std()*np.sqrt(252)),drawdown60=float(1-history[-1,i]/history[-60:,i].max()),volume_ratio=float(self.volume[t,i]/max(self.volume[t-19:t+1,i].mean(),1.)))
   rows.append(r)
  return {'asof':str(self.dates[t]),'initial_capital':float(initial),'cash':float(cash),'equity':equity,'universe':self.tickers.copy(),'stocks':rows,'fees_bps':10,'slippage_bps':5,'max_stock_weight':.10}

def serialize_all(observation):
 rows=observation['stocks'];names=[r['ticker'] for r in rows]
 if names!=observation['universe'] or len(set(names))!=len(names):raise ValueError('Missing/duplicated stock in policy input')
 text=[f"Account asof={observation['asof']} initialINR={observation['initial_capital']:.2f} cashINR={observation['cash']:.2f} equityINR={observation['equity']:.2f} assets={len(rows)}.",
  'All stock rows follow. Quotes are historical market quotes; returns use the original adjusted-price dataset. Whole shares only; no borrowing or shorts. Orders must be funded after costs.',
  'ticker|quoteINR|shares|valueINR|weight|cashAffordableShares|r5|r20|r60|r120|vol20|drawdown60|volumeRatio']
 for r in rows:
  text.append('|'.join([r['ticker'],f"{r['historical_quote']:.2f}",str(r['held_shares']),f"{r['position_value']:.2f}",f"{r['weight']:.4f}",str(r['affordable_shares_cash'])]+[f"{r[k]:.4f}" for k in ['return5','return20','return60','return120','vol20','drawdown60','volume_ratio']]))
 return '\n'.join(text)

def validate_action_coverage(observation,actions):
 if set(actions)!=set(observation['universe']):raise ValueError('Policy must return an explicit action for every dataset ticker')
 if any(not np.isfinite(value) or value<0 for value in actions.values()):raise ValueError('Invalid nonnegative allocation')
