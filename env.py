"""Historical, cost-aware portfolio environment. Research/paper trading only."""
from dataclasses import dataclass
from pathlib import Path
import gymnasium as gym
import numpy as np
import pandas as pd


@dataclass
class FinanceRewardConfig:
    transaction_cost_bps: float = 10.0
    max_weight: float = 0.30
    turnover_penalty: float = 0.001
    volatility_penalty: float = 0.10
    drawdown_penalty: float = 0.20


class PortfolioEnv(gym.Env):
    """Daily, long-only portfolio environment for a local `Date,Ticker,Close` CSV."""
    def __init__(self, csv_path, window=30, initial_cash=100_000.0, reward_config=None,
                 start_date=None, end_date=None):
        super().__init__()
        self.window, self.initial_cash = window, initial_cash
        self.reward_config = reward_config or FinanceRewardConfig()
        self.prices, self.volumes, self.tickers, self.dates = self._load_prices(csv_path)
        mask = np.ones(len(self.dates), dtype=bool)
        if start_date: mask &= self.dates >= np.datetime64(start_date)
        if end_date: mask &= self.dates <= np.datetime64(end_date)
        self.prices, self.volumes, self.dates = self.prices[mask], self.volumes[mask], self.dates[mask]
        self.n_assets = len(self.tickers)
        if len(self.prices) <= window + 2: raise ValueError("CSV needs more rows than the feature window.")
        if self.n_assets * self.reward_config.max_weight < 1: raise ValueError("max_weight must be >= 1 / asset count.")
        self.action_space = gym.spaces.Box(-np.inf, np.inf, shape=(self.n_assets,), dtype=np.float32)
        self.observation_space = gym.spaces.Dict({
            "market": gym.spaces.Box(-np.inf, np.inf, shape=(window, self.n_assets, 5), dtype=np.float32),
            "portfolio": gym.spaces.Box(0, 1, shape=(self.n_assets + 1,), dtype=np.float32),
            "risk": gym.spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32),
        })

    @staticmethod
    def _load_prices(csv_path):
        frame = pd.read_csv(Path(csv_path))
        if missing := {"Date", "Ticker", "Close"} - set(frame.columns): raise ValueError(f"Missing: {sorted(missing)}")
        frame["Date"] = pd.to_datetime(frame["Date"])
        close = frame.pivot(index="Date", columns="Ticker", values="Close").sort_index().ffill().dropna()
        volume = (frame.pivot(index="Date", columns="Ticker", values="Volume").sort_index().reindex(close.index)
                  .reindex(columns=close.columns).fillna(0.0) if "Volume" in frame else close * 0)
        return close.to_numpy(np.float32), volume.to_numpy(np.float32), close.columns.tolist(), close.index.to_numpy()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed); self.index = self.window
        self.weights = np.zeros(self.n_assets, dtype=np.float32); self.cash_weight = 1.0
        self.equity = self.peak_equity = self.initial_cash; self.returns = []
        return self._observation(), {"tickers": self.tickers}

    def step(self, action):
        target = self._target_weights(action); turnover = float(np.abs(target - self.weights).sum())
        cost = turnover * self.reward_config.transaction_cost_bps / 10_000
        gross = float(np.dot(target, self.prices[self.index + 1] / self.prices[self.index] - 1))
        net = gross - cost; self.equity *= 1 + net; self.peak_equity = max(self.peak_equity, self.equity)
        drawdown = 1 - self.equity / self.peak_equity; self.returns.append(net)
        volatility = float(np.std(self.returns[-20:])) if len(self.returns) > 1 else 0.0
        reward = net - self.reward_config.turnover_penalty * turnover - self.reward_config.volatility_penalty * volatility - self.reward_config.drawdown_penalty * drawdown
        self.weights, self.cash_weight, self.index = target, float(max(0, 1 - target.sum())), self.index + 1
        terminated = self.index >= len(self.prices) - 2
        info = {"net_return": net, "gross_return": gross, "turnover": turnover, "trading_cost": cost,
                "equity": self.equity, "drawdown": drawdown, "volatility": volatility}
        return self._observation(), float(reward), terminated, False, info

    def _target_weights(self, action):
        scores = np.asarray(action, dtype=np.float32); scores -= scores.max()
        weights = np.exp(scores) / np.exp(scores).sum(); cap = self.reward_config.max_weight
        for _ in range(self.n_assets):
            over = weights > cap + 1e-7
            if not over.any(): break
            excess = float((weights[over] - cap).sum()); weights[over] = cap; free = ~over
            weights[free] += excess * weights[free] / weights[free].sum()
        return weights.astype(np.float32)

    def _observation(self):
        prices = self.prices[self.index - self.window:self.index + 1]; returns = prices[1:] / prices[:-1] - 1
        log_returns = np.log(prices[1:] / prices[:-1]); rolling_vol = np.array([np.std(returns[max(0, t - 19):t + 1], axis=0) for t in range(len(returns))])
        ma_gap = prices[1:] / np.maximum(np.mean(prices[:-1], axis=0), 1e-6) - 1
        volume = self.volumes[self.index-self.window+1:self.index+1]; volume_z = (volume-volume.mean(0))/(volume.std(0)+1e-6)
        market = np.stack((returns, log_returns, rolling_vol, ma_gap, volume_z), axis=-1).astype(np.float32)
        risk = np.array([np.std(self.returns[-20:]) if len(self.returns)>1 else 0, 1-self.equity/self.peak_equity, self.equity/self.initial_cash-1], dtype=np.float32)
        return {"market": market, "portfolio": np.r_[self.weights, self.cash_weight].astype(np.float32), "risk": risk}
