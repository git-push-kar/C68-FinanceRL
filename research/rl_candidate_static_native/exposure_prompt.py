def prompt(obs,tickers):
    return """Choose a stable long-only stock portfolio allocation strategy. Rebalance every five trading sessions. No leverage or short selling. Trading costs are 10 basis points per unit of stock notional bought or sold. Seek high net portfolio Sharpe with zero risk-free benchmark.
A: Equal weighting across the stock universe.
B: Half equal weighting, half inverse trailing 126-session daily volatility weighting.
C: Half equal weighting, half standardized momentum weighting using 126-session returns excluding the latest 21 sessions.
D: Half equal weighting, half standardized five-session reversal weighting.
All strategy signals use information available one close before execution. This decision sets a stable mixture of strategies, not a market-timing forecast. Respond with exactly A, B, C or D."""
