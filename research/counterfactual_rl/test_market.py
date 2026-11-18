import numpy as np
from research.counterfactual_rl.market import Market,execute,baseline,evaluate,FEE,SLIPPAGE,project_scores

def test_execution_conserves_and_never_borrows():
 rng=np.random.default_rng(5)
 for capital in [500,5000,50000,100000,200000]:
  for _ in range(50):
   price=rng.uniform(10,50000,49);q=rng.integers(0,20,49);w=project_scores(rng.uniform(0,1,49))
   cash,after,f,s,trade=execute(capital,q,price,w)
   assert cash>=0 and np.issubdtype(after.dtype,np.integer) and (after>=0).all()
   assert np.isclose(capital+q@price,cash+after@price+f+s)
   assert np.array_equal(after-q,trade)

def test_fractional_bonus_is_not_truncated_or_duplicated():
 m=Market();t,last=m.bounds('2024-01-01','2024-12-31');day=t+1
 m.open[day]=100;m.close[t:day+1]=100;m.splits[day]=1;m.dividends[day]=0
 m.splits[day,0]=1.5;m.open[day,0]=m.close[day,0]=100/1.5
 q=np.zeros(49,int);q[0]=3;r=m.horizon(t,0,q,None,day)
 assert r['qty'][0]==4 and np.isclose(r['cash'],100/3) and np.isclose(r['values'][0],300)

def test_receivable_counts_in_target_equity_but_cannot_fund_purchases():
 q=np.zeros(49,int);price=np.full(49,100.);w=np.full(49,1/49)
 c,after,*_=execute(0,q,price,w,receivable=10000)
 assert c==0 and after.sum()==0
 c,after,*_=execute(500,q,price,np.r_[.1,np.zeros(48)],receivable=9500)
 assert after[0]<=4 and c>=0

def test_entitlements_survive_rebalance_boundaries_and_are_not_spendable():
 m=Market();t=int(np.searchsorted(m.dates,np.datetime64('2025-01-03')));q=np.zeros(49,int);q[m.tickers.index('ITC.NS')]=17
 one=m.horizon(t,100,q,None,t+1)
 assert np.isclose(one['pending'][1],1.7) and one['cash']==100
 obs=m.observe(t+1,one['cash'],one['qty'],10000,one['pending'])
 assert np.isclose(obs['equity'],100+one['qty']@m.close[t+1]+442)
 listing=int(np.searchsorted(m.dates,np.datetime64('2025-01-29')))
 two=m.horizon(t+1,one['cash'],one['qty'],None,listing,one['pending'])
 full=m.horizon(t,100,q,None,listing)
 assert np.isclose(two['cash'],full['cash']) and np.array_equal(two['qty'],full['qty']) and two['pending'].sum()==0
 assert np.isclose(two['cash'],100+1.7*180*(1-SLIPPAGE)*(1-FEE))

def test_future_execution_prices_do_not_change_observation():
 m=Market();t,_=m.bounds('2024-01-01','2024-12-31');q=np.zeros(49,int);before=m.observe(t,100000,q,100000)
 m.open[t+1:]*=9;m.close[t+1:]*=9;m.prices[t+1:]*=9
 assert m.observe(t,100000,q,100000)==before

def test_buyhold_is_deterministic_and_all49_covered():
 m=Market();a=evaluate(m,baseline('buy_hold'),'2024-01-01','2024-12-31',100000);b=evaluate(m,baseline('buy_hold'),'2024-01-01','2024-12-31',100000)
 assert a==b and len(a['decisions'][0]['trade_shares'])==49
 assert all(sum(abs(x) for x in r['trade_shares'])==0 for r in a['decisions'][1:])

def test_demerged_parent_quote_anchors_and_train_history():
 m=Market()
 for ev in m.events:
  t=int(np.searchsorted(m.dates,np.datetime64(ev['prior'])))
  assert abs(m.close[t,ev['parent_index']]-ev['prior_close'])<1e-6
 first,last=m.bounds('2018-07-02','2023-12-31')
 assert np.isfinite(m.prices[first-126:last+1]).all()
