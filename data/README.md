# Market-data input

Place local historical market-data CSV files here. The training code never
downloads data and never uses broker credentials.

Required format (one row per asset per date):

```csv
Date,Ticker,Close,Volume
2018-01-02,SPY,268.77,86655700
2018-01-02,QQQ,158.49,32540000
```

`Volume` is optional. Dates must be chronological and prices must be adjusted
consistently for splits/dividends before training. Use a chronological training
cutoff and preserve a later test period that is never used for tuning.
