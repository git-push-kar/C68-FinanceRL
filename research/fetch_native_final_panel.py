"""Fetch the already chosen final panel. No policy evaluation or price ranking."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parent
choice_path = ROOT/'NATIVE_FINAL_METHOD_CHOICE.json'
choice = json.loads(choice_path.read_text())
assert choice['test_start'] == '2026-02-02' and choice['test_end'] == '2026-10-30'
assert choice['fixed_universe'] == ['GLD', 'SPY', 'TLT', 'UUP', 'XLE']
assert not (ROOT/'reserved_finance_native_2026_access.json').exists()
out = ROOT/'native_final_data_20261116'
out.mkdir(exist_ok=False)
frames, details = [], {}
for ticker in choice['fixed_universe']:
    h = yf.Ticker(ticker).history(start='2024-01-01', end='2026-10-31',
                                 auto_adjust=False, actions=True, raise_errors=True)
    if h.empty or 'Adj Close' not in h:
        raise RuntimeError('Missing adjusted prices: '+ticker)
    h.to_csv(out/f'{ticker}_raw.csv')
    dates = pd.to_datetime(h.index).tz_localize(None).normalize()
    f = pd.DataFrame({'Date': dates, 'Ticker': ticker,
                      'Close': h['Adj Close'].to_numpy(), 'Volume': h['Volume'].to_numpy()})
    if f.Date.duplicated().any() or not f.Close.gt(0).all() or f.Close.isna().any():
        raise ValueError('Invalid panel values')
    frames.append(f)
    details[ticker] = {'rows': len(f), 'first': str(f.Date.min().date()),
                       'last': str(f.Date.max().date())}
    print(ticker, details[ticker], flush=True)
sets = [set(f.Date) for f in frames]
if any(s != sets[0] for s in sets[1:]):
    raise ValueError('Mismatched sessions; no fill/drop permitted')
panel = pd.concat(frames).sort_values(['Date', 'Ticker'])
if panel.Date.max() != pd.Timestamp(choice['test_end']):
    raise ValueError('Requested final date unavailable; do not substitute another window')
path = out/'five_asset_final_with_warmup.csv'
panel.to_csv(path, index=False)
manifest = {'retrieved_utc': datetime.now(timezone.utc).isoformat(),
            'source': 'Yahoo Finance via yfinance', 'yfinance_version': yf.__version__,
            'method_choice_sha256': hashlib.sha256(choice_path.read_bytes()).hexdigest(),
            'requested': ['2024-01-01', '2026-10-31 exclusive'],
            'evaluation': [choice['test_start'], choice['test_end']],
            'warmup': 'All earlier downloaded rows; no refitting or performance selection',
            'adjustment': 'Vendor Adj Close; raw OHLC, adjusted prices and actions retained',
            'missing_handling': 'Identical complete session sets; no fill or asset dropping',
            'series': details,
            'sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in out.glob('*.csv')},
            'portfolio_outcomes_computed': False}
(out/'manifest.json').write_text(json.dumps(manifest, indent=2))
print('Panel downloaded and hashed; no portfolio outcomes computed.', flush=True)
