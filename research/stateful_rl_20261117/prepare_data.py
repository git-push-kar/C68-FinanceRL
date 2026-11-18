"""Capture a predeclared five-stock Nifty universe and corporate actions."""
from pathlib import Path
import datetime,hashlib,json
import numpy as np
import pandas as pd
import yfinance as yf

ROOT=Path(__file__).resolve().parent
TICKERS=['COALINDIA.NS','HDFCBANK.NS','INFY.NS','SBIN.NS','TCS.NS']

def main():
 out=ROOT/'data_v2';out.mkdir(exist_ok=True)
 if (out/'manifest.json').exists():raise FileExistsError('Completed dataset already exists')
 arrays=[];dates=None;details=[]
 for ticker in TICKERS:
  saved=out/(ticker+'.csv')
  if not saved.exists() and (ROOT/'data'/saved.name).exists():saved.write_bytes((ROOT/'data'/saved.name).read_bytes())
  frame=pd.read_csv(saved,index_col=0,parse_dates=True) if saved.exists() else yf.Ticker(ticker).history(start='2014-01-01',end='2026-11-17',auto_adjust=False,actions=True,repair=False)
  if frame.empty:raise ValueError('Missing '+ticker)
  frame.index=pd.to_datetime(frame.index).tz_localize(None).normalize()
  frame.to_csv(out/(ticker+'.csv'))
  required=['Open','Close','Volume','Dividends','Stock Splits']
  if not all(c in frame for c in required):raise ValueError('Corporate-action columns missing')
  complete=frame.index[frame[required].notna().all(axis=1)]
  if dates is None:dates=complete
  else:dates=dates.intersection(complete)
  arrays.append(frame)
  details.append({'ticker':ticker,'rows':len(frame),'omitted_incomplete_dates':[str(d.date()) for d in frame.index.difference(complete)],'splits':{str(i.date()):float(v) for i,v in frame['Stock Splits'].items() if v},'dividend_events':int((frame.Dividends!=0).sum())})
 if len(dates)<2500:raise ValueError('Unexpectedly short shared history')
 opened=[];closed=[];divs=[];split=[];volumes=[]
 for f in arrays:
  f=f.loc[dates];ratios=f['Stock Splits'].to_numpy(float);ratios=np.where(ratios==0,1.,ratios)
  # Yahoo historical OHLC and dividends are split adjusted even when auto_adjust=False.
  # Undo subsequent splits to recover historical share denominations. Future split
  # records repair source units only; no future event is included in model observations.
  future=np.cumprod(ratios[::-1])[::-1]/ratios
  opened.append(f.Open.to_numpy(float)*future);closed.append(f.Close.to_numpy(float)*future)
  divs.append(f.Dividends.to_numpy(float)*future);split.append(ratios);volumes.append(f.Volume.to_numpy(float))
 values={'dates':dates.to_numpy(dtype='datetime64[D]'),'tickers':np.array(TICKERS),'open':np.array(opened).T,'close':np.array(closed).T,'dividends':np.array(divs).T,'splits':np.array(split).T,'volume':np.array(volumes).T}
 for name in ['open','close','dividends','splits','volume']:
  if not np.isfinite(values[name]).all():raise ValueError('Nonfinite '+name)
 if (values['open']<=0).any() or (values['close']<=0).any():raise ValueError('Nonpositive price')
 # These selected stocks have integer forward splits in this capture. Reject an
 # unexpected fractional/reverse event instead of inventing cash-in-lieu accounting.
 if not np.allclose(values['splits'],np.round(values['splits'])) or (values['splits']<1).any():raise ValueError('Fractional corporate action requires explicit review')
 np.savez_compressed(out/'market.npz',**values)
 manifest={'captured_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source':'Yahoo Finance via yfinance '+yf.__version__,'tickers':TICKERS,'universe_reason':'Five long-established liquid Nifty constituents: coal, banks and IT. Initial candidate RELIANCE replaced by TCS before any policy evaluation because the Jio demerger entitlement is not modeled by Yahoo cash-dividend/split events. No asset ranking by test returns. Sector concentration and fixed-universe survivorship remain limitations. Prior unselected capture retained in data/.','start':str(dates[0].date()),'end':str(dates[-1].date()),'rows':len(dates),'details':details,'unit_reconstruction':'OHLC and cash dividends multiplied by the product of subsequent Yahoo split factors, recovering the then-current share denomination. No bfill/interpolation. Shared observed sessions only.','dividend_assumption':'Cash credited on recorded ex-date rather than actual payment date, equally for all policies. Taxes and impact omitted.','sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.is_file()}}
 (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
 print(json.dumps({'rows':len(dates),'end':manifest['end'],'details':details},indent=2),flush=True)

if __name__=='__main__':main()
