"""Immutable full-universe inventory and execution-data reconciliation.

No policy evaluation, asset selection, price repair or source rewriting.
"""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
SOURCE=ROOT.parent/'data/nifty50_prices.csv'
EXPECTED='29b847a33ad88b7568eef66e1fb9f3274a897bd3091ddb8551c129b75dd726f7'

def main():
 digest=hashlib.sha256(SOURCE.read_bytes()).hexdigest()
 if digest!=EXPECTED:raise ValueError('Original dataset changed')
 frame=pd.read_csv(SOURCE,parse_dates=['Date'])
 if frame.duplicated(['Date','Ticker']).any():raise ValueError('Duplicate source rows')
 tickers=sorted(frame.Ticker.unique());assert len(tickers)==49
 price=frame.pivot(index='Date',columns='Ticker',values='Close').sort_index().reindex(columns=tickers)
 volume=frame.pivot(index='Date',columns='Ticker',values='Volume').reindex(index=price.index,columns=tickers)
 rows=[]
 for ticker in tickers:
  original=frame[frame.Ticker==ticker].set_index('Date').sort_index()
  raw=pd.read_csv(ROOT/'execution_fields'/f'{ticker}.csv',index_col=0)
  raw.index=pd.to_datetime(raw.index,utc=True).tz_convert('Asia/Kolkata').tz_localize(None).normalize()
  if raw.index.duplicated().any():raise ValueError('Duplicate supplementary dates '+ticker)
  common=original.index.intersection(raw.index)
  joined=pd.DataFrame({'original':original.Close.reindex(common),'supplement_adj':raw['Adj Close'].reindex(common),'supplement_close':raw.Close.reindex(common)})
  rets=joined[['original','supplement_adj']].pct_change(fill_method=None)
  difference=(rets.original-rets.supplement_adj).abs();large=difference[difference>.005]
  split=raw['Stock Splits'];noninteger=split[(split!=0)&(abs(split-split.round())>1e-6)]
  missing=original.index[(original.index>=pd.Timestamp('2018-01-01'))&~original.index.isin(raw.index)]
  rows.append({'ticker':ticker,'source_rows':len(original),'first':str(original.index.min().date()),'last':str(original.index.max().date()),'supplement_symbol':'LTM.NS' if ticker=='LTIM.NS' else ticker,'missing_supplement_dates_since2018':[str(d.date()) for d in missing],
   'return_discrepancies_over50bps':[{'date':str(d.date()),'absolute_return_difference':float(v)} for d,v in large.items()],
   'fractional_split_events':[{'date':str(d.date()),'ratio':float(v)} for d,v in noninteger.items()]})
 coverage=[]
 for year in range(1999,2027):
  block=price[price.index.year==year]
  if len(block):coverage.append({'year':year,'sessions':len(block),'tickers_with_any_observation':int(block.notna().any().sum()),'sessions_all49_observed':int(block.notna().all(axis=1).sum()),'missing_cells':int(block.isna().sum().sum())})
 out=ROOT/'original_panel.npz'
 np.savez_compressed(out,dates=price.index.to_numpy(dtype='datetime64[D]'),tickers=np.array(tickers),adjusted_close=price.to_numpy(float),volume=volume.to_numpy(float),observed=price.notna().to_numpy())
 report={'original_sha256':digest,'source_rows':len(frame),'asset_count':len(tickers),'tickers':tickers,'first_date':str(price.index.min().date()),'last_date':str(price.index.max().date()),'full_history_retained':True,'fills_applied':False,'assets_dropped':[],'coverage':coverage,'series_audit':rows,
 'panel_sha256':hashlib.sha256(out.read_bytes()).hexdigest(),
 'interpretation':'Original adjusted prices preserved exactly; not historical executable quotes. Supplementary data must not silently replace source observations. Return discrepancies and demerger accounting require explicit review before claiming realistic whole-share fills. Pre-listing cells remain missing, not invented.',
 'known_corporate_events_requiring_review':['RELIANCE July2023 demerger; historical vendor OHLC may already embed adjustments','ITC January2025 hotels demerger; entitlement and pre-event quote reconstruction','fractional bonus allocations: explicit cash-in-lieu handling required'],
 'symbol_alias_source':'https://nsearchives.nseindia.com/content/circulars/CML72948.pdf'}
 (ROOT/'original_dataset_audit.json').write_text(json.dumps(report,indent=2))
 print(json.dumps({'assets':len(tickers),'rows':len(frame),'dates':[report['first_date'],report['last_date']],'recent_coverage':[r for r in coverage if r['year']>=2018],'discrepancy_counts':{r['ticker']:len(r['return_discrepancies_over50bps']) for r in rows if r['return_discrepancies_over50bps']}},indent=2))

if __name__=='__main__':main()
