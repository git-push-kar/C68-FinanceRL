"""Validate provider overlap; never evaluate a portfolio on new dates."""
import hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd

root=Path(__file__).parent;folder=root/'data_extension_20261107'
manifest=json.loads((folder/'download_manifest.json').read_text(encoding='utf-8-sig'))
old=pd.read_csv(root/'data/nifty50_prices.csv');tickers=sorted(old.Ticker.unique())
if len(manifest)!=len(tickers) or any(x['status']!='downloaded' for x in manifest):
    raise RuntimeError('Incomplete universe; do not drop failures')
rows=[];checks=[]
for ticker in tickers:
    path=folder/(ticker+'.json');raw=json.loads(path.read_text())['chart']['result'][0]
    adj=raw['indicators']['adjclose'][0]['adjclose'];quote=raw['indicators']['quote'][0]
    dates=pd.to_datetime(raw['timestamp'],unit='s',utc=True).tz_convert('Asia/Kolkata').strftime('%Y-%m-%d')
    new=pd.DataFrame({'Date':dates,'Ticker':ticker,'Close':adj,'Volume':quote['volume']})
    if new.Date.duplicated().any():raise ValueError('Duplicate dates '+ticker)
    new=new[new.Date<='2026-10-30'].copy()
    if new[['Close','Volume']].isna().any().any() or (new.Close<=0).any():raise ValueError('Invalid prices '+ticker)
    prior=old[old.Ticker==ticker].copy();overlap=prior.merge(new,on=['Date','Ticker'])
    ratio=overlap.Close_y/overlap.Close_x;factor=float(ratio.median());error=float(np.max(abs(ratio/factor-1)))
    checks.append({'ticker':ticker,'overlap_rows':len(overlap),'historical_scale_factor':factor,
                   'maximum_relative_overlap_error':error,'new_last_date':new.Date.max()})
    if len(overlap)<100 or error>1e-4:raise ValueError(f'Adjustment mismatch {ticker}: {checks[-1]}')
    prior.Close*=factor
    rows.extend([prior,new[new.Date>prior.Date.max()]])
combined=pd.concat(rows).sort_values(['Date','Ticker'])
future=combined[combined.Date>'2026-01-30'];counts=future.groupby('Date').Ticker.nunique()
if not len(counts) or (counts!=len(tickers)).any():raise ValueError('Incomplete new date panel')
target=folder/'nifty49_adjusted_extended.csv'
if target.exists():raise FileExistsError(target)
combined.to_csv(target,index=False)
report={'status':'passed','universe':tickers,'original_unchanged':True,'source':'Yahoo chart adjusted close; raw responses retained',
    'transformation':'Per-ticker constant rescaling of old history to match current adjusted close; new dates appended. No historical return changes except floating-point roundoff.',
    'checks':checks,'new_first_date':future.Date.min(),'new_last_date':future.Date.max(),'new_sessions':len(counts),
    'dataset_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'strategy_evaluation_performed':False,
    'limits':'Provider data not exchange-certified; fixed historical universe remains survivor-selected. Download and adjustment evidence do not establish investment performance.'}
(folder/'VALIDATION.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k not in ('checks','universe')},indent=2))
