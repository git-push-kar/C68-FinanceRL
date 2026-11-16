"""Data-quality checks restricted to training/development; no test outcomes."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

path=Path('data/nifty50_prices.csv')
frame=pd.read_csv(path,parse_dates=['Date'])
metadata={'dataset_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
    'raw_rows':len(frame),'tickers':sorted(frame.Ticker.unique().tolist()),
    'date_coverage_metadata':[str(frame.Date.min().date()),str(frame.Date.max().date())]}
frame=frame[frame.Date<='2024-12-31'].copy()
metadata.update(duplicate_date_tickers=int(frame.duplicated(['Date','Ticker']).sum()),
    nonpositive_close=int((frame.Close<=0).sum()),nonfinite_close=int((~np.isfinite(frame.Close)).sum()),
    missing_close=int(frame.Close.isna().sum()))
prices=frame.pivot(index='Date',columns='Ticker',values='Close').sort_index()
filled=prices.ffill().dropna()
returns=filled.pct_change(fill_method=None).iloc[1:]
extreme=returns.stack()
extreme=extreme[extreme.abs()>.30]
metadata.update(missing_panel_cells_before_ffill=int(prices.isna().sum().sum()),
    effective_training_development_dates=[str(filled.index.min().date()),str(filled.index.max().date())],
    effective_dates=len(filled),extreme_abs_daily_moves_over_30pct=[
        {'date':str(date.date()),'ticker':ticker,'return':float(value)} for (date,ticker),value in extreme.items()],
    training_dates=int((filled.index<='2022-12-31').sum()),
    development_dates=int(((filled.index>='2023-01-01')&(filled.index<='2024-12-31')).sum()),
    limitations=['Fixed universe selected after historical period; membership/survivorship bias remains.',
        'Bundled data provenance is historical; current downloader uses adjusted Yahoo closes but cannot retroactively prove the original download settings.',
        'Price/data-quality audit is not a final-test performance evaluation. No 2025 returns computed.'])
output=Path('campaign_runs/DATA_QUALITY.json');output.write_text(json.dumps(metadata,indent=2,allow_nan=False))
print(json.dumps(metadata,indent=2))
