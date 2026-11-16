import hashlib,json
from pathlib import Path
import pandas as pd

folder=Path(__file__).parent/'sector_data_20261107'
manifest=json.loads((folder/'download_manifest.json').read_text(encoding='utf-8-sig'))
if len(manifest)!=9 or any(r['status']!='downloaded' for r in manifest):raise ValueError('Incomplete universe')
parts=[];coverage=[]
for record in manifest:
    path=folder/(record['ticker']+'.json')
    if hashlib.sha256(path.read_bytes()).hexdigest().upper()!=record['sha256']:raise ValueError('Hash mismatch')
    raw=json.loads(path.read_text())['chart']['result'][0];quote=raw['indicators']['quote'][0]
    dates=pd.to_datetime(raw['timestamp'],unit='s',utc=True).tz_convert('America/New_York').strftime('%Y-%m-%d')
    frame=pd.DataFrame({'Date':dates,'Ticker':record['ticker'],'Close':raw['indicators']['adjclose'][0]['adjclose'],'Volume':quote['volume']})
    if frame.Date.duplicated().any() or frame[['Close','Volume']].isna().any().any() or (frame.Close<=0).any():raise ValueError('Invalid records')
    parts.append(frame);coverage.append({'ticker':record['ticker'],'first':frame.Date.min(),'last':frame.Date.max(),'rows':len(frame)})
data=pd.concat(parts).sort_values(['Date','Ticker']);counts=data.groupby('Date').Ticker.nunique()
if (counts!=9).any():raise ValueError('Common panel has missing observations')
# Training drivers only receive the explicit development file.
target=folder/'sector9_development_through2024.csv'
if target.exists():raise FileExistsError(target)
data[data.Date<='2024-12-31'].to_csv(target,index=False)
report={'status':'passed','coverage':coverage,'missing_common_panel_cells':0,
    'development_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),
    'raw_later_dates_retained_but_not_exported_to_training_driver':True,
    'portfolio_evaluation_performed':False,'source':'Yahoo adjusted closes; raw dividends/splits retained',
    'limits':'Provider data not exchange-certified; fixed original sector universe, not a point-in-time all-sector investable index.'}
(folder/'VALIDATION.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
