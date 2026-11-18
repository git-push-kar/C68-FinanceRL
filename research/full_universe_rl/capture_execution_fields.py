"""Supplement the exact original49 tickers; never replace/drop the universe."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import json,hashlib,datetime
import pandas as pd
import yfinance as yf

ROOT=Path(__file__).resolve().parent
SOURCE=ROOT.parent/'data/nifty50_prices.csv'

def main():
 frame=pd.read_csv(SOURCE);tickers=sorted(frame.Ticker.unique());assert len(tickers)==49
 out=ROOT/'execution_fields';out.mkdir(exist_ok=True)
 protocol={'original_file':str(SOURCE),'original_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),'tickers':tickers,'original_last_date':frame.Date.max(),'supplement_only':'Unadjusted OHLC,corporate events for historical share denomination and actual affordability. Original CSV remains immutable. No new stocks and no extension of primary evaluation dates.','requested_start':'2014-01-01','requested_end_exclusive':'2026-11-18','reason_later_events':'Yahoo historical OHLC are split adjusted even withauto_adjustFalse; later splits are needed only to reconstruct historical share denominations. No later returns enter original-date observations/evaluation.'}
 (out/'capture_protocol.json').write_text(json.dumps(protocol,indent=2))
 def fetch(ticker):
  path=out/(ticker+'.csv')
  if path.exists():h=pd.read_csv(path,index_col=0,parse_dates=True)
  else:
   symbol='LTM.NS' if ticker=='LTIM.NS' else ticker
   h=yf.Ticker(symbol).history(start='2014-01-01',end='2026-11-18',auto_adjust=False,actions=True,repair=False,raise_errors=True)
   if h.empty:raise ValueError('Empty raw capture '+ticker)
   h.to_csv(path)
  required=['Open','Close','Dividends','Stock Splits']
  if not all(k in h for k in required):raise ValueError('Incomplete raw fields '+ticker)
  return {'ticker':ticker,'rows':len(h),'first':str(h.index.min()),'last':str(h.index.max()),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
 results=[];errors=[]
 with ThreadPoolExecutor(max_workers=4) as executor:
  jobs={executor.submit(fetch,t):t for t in tickers}
  for job in as_completed(jobs):
   try:r=job.result();results.append(r);print(json.dumps(r),flush=True)
   except Exception as exc:errors.append({'ticker':jobs[job],'error':repr(exc)});print(json.dumps(errors[-1]),flush=True)
 report={'captured_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'results':results,'errors':errors,'all_original_tickers_captured':not errors and len(results)==len(tickers)}
 (out/'capture_receipt.json').write_text(json.dumps(report,indent=2))
 if errors:raise RuntimeError('Do not drop failed tickers; resolve capture errors')

if __name__=='__main__':main()
