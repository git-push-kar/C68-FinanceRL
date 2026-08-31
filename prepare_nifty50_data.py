"""Download and normalize the current official Nifty 50 universe for training.

This script is for historical research.  It uses today's constituent list, so
long historical backtests contain survivorship bias.  Use NSE point-in-time
constituent archives before treating results as investment evidence.
"""
import argparse
from pathlib import Path

import pandas as pd
import yfinance as yf


# Linked as "Download List of Nifty 50 stocks (.csv)" on the official NSE page.
NIFTY50_CONSTITUENTS_URL = "https://nsearchives.nseindia.com/content/indices/ind_nifty50list.csv"


def current_nifty50_symbols():
    constituents = pd.read_csv(NIFTY50_CONSTITUENTS_URL)
    if "Symbol" not in constituents.columns:
        raise RuntimeError(f"Unexpected NSE constituent CSV columns: {constituents.columns.tolist()}")
    return constituents["Symbol"].astype(str).str.strip().add(".NS").tolist()


def main():
    parser = argparse.ArgumentParser(description="Create Finance RL market panel from current Nifty 50 symbols.")
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default="2025-01-01", help="Exclusive end date in YYYY-MM-DD form")
    parser.add_argument("--output", default="data/nifty50_prices.csv")
    args = parser.parse_args()

    tickers = current_nifty50_symbols()
    print(f"Downloading {len(tickers)} current Nifty 50 symbols from {args.start} through {args.end}...")
    raw = yf.download(tickers=tickers, start=args.start, end=args.end, auto_adjust=True,
                      group_by="column", threads=True, progress=False)
    if raw.empty or "Close" not in raw.columns:
        raise RuntimeError("Yahoo Finance returned no usable closing-price data. Check your network/date range.")
    close = raw["Close"]
    volume = raw["Volume"] if "Volume" in raw.columns else pd.DataFrame(0, index=close.index, columns=close.columns)
    if isinstance(close, pd.Series):
        close, volume = close.to_frame(name=tickers[0]), volume.to_frame(name=tickers[0])

    records = []
    for ticker in close.columns:
        frame = pd.DataFrame({"Date": close.index, "Ticker": ticker,
                              "Close": close[ticker].to_numpy(), "Volume": volume[ticker].to_numpy()})
        records.append(frame.dropna(subset=["Close"]))
    panel = pd.concat(records, ignore_index=True).sort_values(["Date", "Ticker"])
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(output, index=False)
    print(f"Wrote {len(panel):,} rows for {panel['Ticker'].nunique()} symbols to {output}")
    print("Reminder: use an earlier train cutoff and a later untouched test period.")


if __name__ == "__main__":
    main()
