"""Fixed literature-motivated universe; no prices after2024 requested."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json
import pandas as pd
import yfinance as yf
R=Path(__file__).parent;out=R/'multiasset_data_20261115';out.mkdir(exist_ok=False)
tickers=['GLD','SPY','TLT','UUP','XLE'];frames=[];details={}
for ticker in tickers:
 h=yf.Ticker(ticker).history(start='2008-01-01',end='2025-01-01',auto_adjust=False,actions=True,raise_errors=True)
 if h.empty or 'Adj Close' not in h:raise RuntimeError('Missing adjusted prices: '+ticker)
 h.to_csv(out/f'{ticker}_raw.csv');dates=pd.to_datetime(h.index).tz_localize(None).normalize()
 f=pd.DataFrame({'Date':dates,'Ticker':ticker,'Close':h['Adj Close'].to_numpy(),'Volume':h['Volume'].to_numpy()})
 if f.Date.duplicated().any() or not f.Close.gt(0).all() or f.Close.isna().any():raise ValueError('Invalid prices')
 if f.Date.max()>pd.Timestamp('2024-12-31'):raise ValueError('Reserved dates included')
 frames.append(f);details[ticker]={'rows':len(f),'first':str(f.Date.min().date()),'last':str(f.Date.max().date())}
 print(ticker,details[ticker],flush=True)
sets=[set(f.Date) for f in frames]
if any(d!=sets[0] for d in sets[1:]):raise ValueError('Missing/mismatched sessions; explicit review required, no silent fill/drop')
combined=pd.concat(frames).sort_values(['Date','Ticker']);path=out/'five_asset_development_through2024.csv';combined.to_csv(path,index=False)
manifest={'source':'Yahoo Finance via yfinance','retrieved_utc':datetime.now(timezone.utc).isoformat(),'yfinance_version':yf.__version__,'fixed_universe':tickers,'selection_rationale':'FinATOM five macro exposures, fixed before download; not performance selected','dates_requested':['2008-01-01','2025-01-01 exclusive'],'adjustment':'Yahoo Adj Close, vendor adjusted; raw OHLC/actions preserved','missing_handling':'Require identical date sets; no filling or asset dropping','series':details,'sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.glob('*.csv')},'test2026_accessed':False}
(out/'manifest.json').write_text(json.dumps(manifest,indent=2));print('VALIDATED',len(sets[0]),'shared sessions',flush=True)
