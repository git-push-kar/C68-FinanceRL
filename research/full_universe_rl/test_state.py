import numpy as np
import pytest
from state import OriginalPanel,serialize_all,validate_action_coverage

def fixture():
 p=OriginalPanel();t=int(np.searchsorted(p.dates,np.datetime64('2024-01-02')))
 # Explicit artificial quotes for contract tests, not financial evaluation.
 quotes=np.linspace(100,5000,49);shares=np.arange(49,dtype=np.int64)
 return p,t,quotes,shares

def test_all49_account_rows_and_actions_required():
 p,t,q,s=fixture();o=p.observe(t,10000.,s,100000.,q);text=serialize_all(o)
 assert len(o['stocks'])==49 and all(ticker in text for ticker in p.tickers)
 assert o['equity']==10000+s@q
 actions={ticker:1/49 for ticker in p.tickers};validate_action_coverage(o,actions)
 del actions[p.tickers[-1]]
 with pytest.raises(ValueError):validate_action_coverage(o,actions)

def test_future_data_cannot_change_observation():
 p,t,q,s=fixture();before=p.observe(t,10000.,s,100000.,q)
 p.prices[t+1:]*=10;p.volume[t+1:]*=100
 assert p.observe(t,10000.,s,100000.,q)==before

def test_capital_and_holdings_are_visible_and_require_complete_universe():
 p,t,q,s=fixture();a=p.observe(t,500.,s,100000.,q);b=p.observe(t,5000.,s,100000.,q)
 assert serialize_all(a)!=serialize_all(b)
 assert b['stocks'][0]['affordable_shares_cash']>a['stocks'][0]['affordable_shares_cash']
 with pytest.raises(ValueError):p.observe(t,500.,s[:5],100000.,q[:5])

def test_original_source_values_are_preserved():
 import pandas as pd
 from pathlib import Path
 p=OriginalPanel();d=pd.read_csv(Path(__file__).parent.parent/'data/nifty50_prices.csv')
 for row in d.sample(100,random_state=42).itertuples():
  t=int(np.searchsorted(p.dates,np.datetime64(row.Date)));i=p.tickers.index(row.Ticker)
  assert p.prices[t,i]==row.Close
