import copy
import numpy as np
from market import Market,execute,targets,ACTIONS,evaluate,baseline,prompt

def test_integer_feasibility_and_conservation():
 rng=np.random.default_rng(29)
 for _ in range(50):
  price=rng.uniform(100,5000,5);qty=rng.integers(0,20,5);cash=float(rng.uniform(0,10000));w=rng.dirichlet(np.ones(6))[:5]
  c,q,f,s=execute(cash,qty,price,w)
  assert q.dtype==np.int64 and (q>=0).all() and (c>=0).all()
  assert np.allclose(c+(q*price).sum(1)+f+s,cash+qty@price)

def test_small_account_cannot_buy_expensive_stock():
 c,q,f,s=execute(500,np.zeros(5,int),np.array([1000.,2000,3000,4000,5000]),np.array([1,0,0,0,0]))
 assert c[0]==500 and q.sum()==0 and f[0]==0
 c,q,_,_=execute(5000,np.zeros(5,int),np.array([1000.,2000,3000,4000,5000]),np.array([1,0,0,0,0]),enforce_risk=False)
 assert q[0,0]==4 and c[0]>0

def test_keep_means_keep_shares_even_after_price_gap():
 m=Market();t,_=m.bounds('2025-01-01','2025-02-01');q=(10000/m.close[t]).astype(np.int64);cash=5000.
 w=targets(np.zeros(5,int),q,cash,m.close[t]);out=m.horizon(t,cash,q,w,t+5,keep=np.ones(5,bool))
 assert out['fees'][0]==0 and not out['trade_shares'].any()
 assert np.array_equal(out['qty'][0],q*np.prod(m.splits[t+1:t+6],axis=0))

def test_vectorized_all_actions_match_individual_execution():
 m=Market();t,_=m.bounds('2020-01-01','2020-12-31');q=np.array([2,4,0,3,1]);cash=12000.
 w=targets(ACTIONS,q,cash,m.close[t]);all_result=m.horizon(t,cash,q,w,t+5,keep=ACTIONS==0)
 for i in [0,1,71,625,7777,16806]:
  one=m.horizon(t,cash,q,w[i],t+5,keep=ACTIONS[i]==0)
  assert np.allclose(all_result['values'][:,i],one['values'][:,0],atol=1e-8)
  assert np.array_equal(all_result['qty'][i],one['qty'][0])

def test_observation_is_unchanged_by_future_mutation():
 m=Market();t,_=m.bounds('2021-01-01','2021-06-01');q=np.ones(5,int);before=prompt(m.observe(t,1000.,q,50000))
 m.close[t+1:]*=10;m.open[t+1:]*=.1;m.volume[t+1:]*=100;m.relative[t+1:]=100;m.index[t+1:]=1
 assert prompt(m.observe(t,1000.,q,50000))==before

def test_buy_hold_never_rebalances_after_initial_purchase():
 m=Market();out=evaluate(m,baseline('buy_hold'),'2025-01-01','2026-11-13',100000)
 assert not any(any(r['trade_shares']) for r in out['decisions'][1:])
 assert out['metrics']['fees']>0

def test_split_conserves_economic_value_and_never_trades_for_hold():
 m=Market();t=int(np.searchsorted(m.dates,np.datetime64('2025-08-26')))-1
 assert m.splits[t+1,1]==2
 q=np.array([0,10,0,0,0]);out=m.horizon(t,0.,q,None,t+1)
 assert out['qty'][0,1]==20 and out['fees'][0]==0
 expected=20*m.close[t+1,1]+20*m.dividends[t+1,1]
 assert np.isclose(out['values'][0,0],expected)

def test_all_required_stock_and_account_facts_are_present():
 m=Market();t,_=m.bounds('2025-01-01','2025-06-01');o=m.observe(t,500.,np.zeros(5,int),500.)
 text=prompt(o)
 for name in m.tickers:assert name in text
 assert 'available cash 500.00' in text and 'Initial capital 500.00' in text
 assert 'shares affordable' in text and all('price' in s and 'shares' in s and 'vol20' in s for s in o['stocks'])

def test_concentration_limits_are_satisfied_after_fees():
 from market import concentration_limits,SECTORS
 rng=np.random.default_rng(44)
 for _ in range(100):
  price=rng.uniform(100,5000,5);qty=rng.integers(0,30,5);cash=float(rng.uniform(500,200000));equity=cash+qty@price
  stock_cap,sector_cap=concentration_limits(np.array([equity]),price)
  for keep in [None,np.ones(5,bool)]:
   c,q,_,_=execute(cash,qty,price,rng.dirichlet(np.ones(5)),keep=keep)
   after=float(c[0]+q[0]@price)
   assert np.max(q[0]*price/after)<=stock_cap[0]+1e-8
   assert max(float(q[0,g]@price[g]/after) for g in SECTORS)<=sector_cap[0]+1e-8

def test_small_account_limit_allows_its_only_affordable_stock():
 price=np.array([400.,1000,2000,3000,4000]);c,q,_,_=execute(500,np.zeros(5,int),price,np.array([1,0,0,0,0]))
 assert q[0,0]==1 and c[0]>0

def test_single_stock_holding_is_not_weakened_by_active_risk_limits():
 m=Market();out=evaluate(m,baseline('single_hold_0'),'2025-01-01','2025-06-30',100000.)
 first=out['decisions'][0];q=first['shares_after_horizon'];assert sum(v>0 for v in q)==1
 assert first['cash_after_horizon']<10000
 assert not any(any(r['trade_shares']) for r in out['decisions'][1:])
