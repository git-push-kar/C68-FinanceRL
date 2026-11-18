"""Historical, cost-aware portfolio environment. Research/paper trading only."""
from dataclasses import dataclass
from pathlib import Path
import gymnasium as gym
import numpy as np
import pandas as pd
from allocation import allocation_weights


@dataclass
class FinanceRewardConfig:
    transaction_cost_bps: float = 10.0
    max_weight: float = 0.30
    turnover_penalty: float = 0.001
    volatility_penalty: float = 0.10
    drawdown_penalty: float = 0.20
    rebalance_rate: float = 1.0
    reward_mode: str = "legacy_risk"
    reward_scale: float = 1.0
    risk_aversion: float = 20.0
    rebalance_interval: int = 1

    def __post_init__(self):
        if not isinstance(self.rebalance_interval, int) or self.rebalance_interval < 1:
            raise ValueError('Rebalance interval must be a positive integer')
        if not np.isfinite(self.rebalance_rate) or not 0 < self.rebalance_rate <= 1:
            raise ValueError("Rebalance rate must be in (0, 1]")
        if self.reward_mode not in {"legacy_risk", "relative_log", "relative_mean_variance"}:
            raise ValueError("Unknown finance reward objective")
        if not np.isfinite(self.reward_scale) or self.reward_scale <= 0:
            raise ValueError("Reward scale must be finite and positive")
        if not np.isfinite(self.risk_aversion) or self.risk_aversion < 0:
            raise ValueError('Risk aversion must be finite and nonnegative')


class PortfolioEnv(gym.Env):
    """Daily, long-only portfolio environment for a local `Date,Ticker,Close` CSV."""
    def __init__(self, csv_path, window=30, initial_cash=100_000.0, reward_config=None,
                 start_date=None, end_date=None, covariance_window=0):
        super().__init__()
        self.window, self.initial_cash = window, initial_cash
        self.reward_config = reward_config or FinanceRewardConfig()
        self.covariance_window = covariance_window
        self._covariance_cache = {}
        self.prices, self.volumes, self.tickers, self.dates = self._load_prices(csv_path)
        # Keep pre-start history for features. start_date selects the first
        # return-ending date; its allocation is chosen at the preceding close.
        mask = np.ones(len(self.dates), dtype=bool)
        if end_date: mask &= self.dates <= np.datetime64(end_date)
        self.prices, self.volumes, self.dates = self.prices[mask], self.volumes[mask], self.dates[mask]
        self.n_assets = len(self.tickers)
        if len(self.prices) <= window + 2: raise ValueError("CSV needs more rows than the feature window.")
        self.start_index = max(window, covariance_window)
        if start_date:
            first_return = int(np.searchsorted(self.dates, np.datetime64(start_date)))
            self.start_index = first_return - 1
            if self.start_index < max(window, covariance_window):
                raise ValueError("Not enough pre-start history for the feature window.")
            if first_return >= len(self.prices):
                raise ValueError("No return dates in the requested period.")
        if self.n_assets * self.reward_config.max_weight < 1: raise ValueError("max_weight must be >= 1 / asset count.")
        self.action_space = gym.spaces.Box(-np.inf, np.inf, shape=(self.n_assets,), dtype=np.float32)
        self.observation_space = gym.spaces.Dict({
            "market": gym.spaces.Box(-np.inf, np.inf, shape=(window, self.n_assets, 5), dtype=np.float32),
            "portfolio": gym.spaces.Box(0, 1, shape=(self.n_assets + 1,), dtype=np.float32),
            "risk": gym.spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32),
        })
        if covariance_window:
            self.observation_space['covariance'] = gym.spaces.Box(-np.inf,np.inf,
                shape=(self.n_assets,self.n_assets),dtype=np.float32)

    @staticmethod
    def _load_prices(csv_path):
        frame = pd.read_csv(Path(csv_path))
        if missing := {"Date", "Ticker", "Close"} - set(frame.columns): raise ValueError(f"Missing: {sorted(missing)}")
        frame["Date"] = pd.to_datetime(frame["Date"])
        close = frame.pivot(index="Date", columns="Ticker", values="Close").sort_index().ffill().dropna()
        volume = (frame.pivot(index="Date", columns="Ticker", values="Volume").sort_index().reindex(close.index)
                  .reindex(columns=close.columns).fillna(0.0) if "Volume" in frame else close * 0)
        prices = close.to_numpy(np.float64)
        if not np.isfinite(prices).all() or (prices <= 0).any():
            raise ValueError("Prices must be finite and strictly positive.")
        return prices, volume.to_numpy(np.float64), close.columns.tolist(), close.index.to_numpy()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed); self.index = self.start_index
        self.weights = np.zeros(self.n_assets, dtype=np.float32); self.cash_weight = 1.0
        self.equity = self.peak_equity = self.initial_cash; self.returns = []
        self.benchmark_weights = np.zeros(self.n_assets, dtype=np.float64)
        return self._observation(), {"tickers": self.tickers}

    def step(self, action):
        if self.index >= len(self.prices) - 1:
            raise RuntimeError("Episode finished; call reset before step.")
        if (self.index-self.start_index)%self.reward_config.rebalance_interval:
            target = self.weights.copy()
        else:
            target = self._target_weights(action)
        turnover = float(np.abs(target - self.weights).sum())
        cost = turnover * self.reward_config.transaction_cost_bps / 10_000
        relatives = self.prices[self.index + 1] / self.prices[self.index]
        gross = float(np.dot(target, relatives - 1))
        # Cost is charged on traded asset notional before the market return.
        net = (1 - cost) * (1 + gross) - 1
        benchmark_target = np.full(self.n_assets, 1 / self.n_assets)
        benchmark_turnover = float(np.abs(benchmark_target - self.benchmark_weights).sum())
        benchmark_cost = benchmark_turnover * self.reward_config.transaction_cost_bps / 10_000
        benchmark_gross = float(np.dot(benchmark_target, relatives - 1))
        benchmark_net = (1 - benchmark_cost) * (1 + benchmark_gross) - 1
        self.benchmark_weights = benchmark_target * relatives / (1 + benchmark_gross)
        self.equity *= 1 + net; self.peak_equity = max(self.peak_equity, self.equity)
        drawdown = 1 - self.equity / self.peak_equity; self.returns.append(net)
        volatility = float(np.std(self.returns[-20:])) if len(self.returns) > 1 else 0.0
        turnover_term = self.reward_config.turnover_penalty * turnover
        volatility_term = self.reward_config.volatility_penalty * volatility
        drawdown_term = self.reward_config.drawdown_penalty * drawdown
        if self.reward_config.reward_mode == 'relative_mean_variance':
            coefficient = .5 * self.reward_config.risk_aversion
            reward = self.reward_config.reward_scale * (net - benchmark_net - coefficient * (net * net - benchmark_net * benchmark_net))
        elif self.reward_config.reward_mode == 'relative_log':
            # Additive log excess wealth, with actual trading costs on both sides.
            # Risk is assessed separately by validation drawdown/Sharpe; no old
            # drawdown level is repeatedly subtracted from this objective.
            reward = self.reward_config.reward_scale * (np.log1p(net) - np.log1p(benchmark_net))
        else:
            reward = self.reward_config.reward_scale * (net - turnover_term - volatility_term - drawdown_term)
        decision_date = str(self.dates[self.index])[:10]
        self.weights = target * relatives / (1 + gross)
        self.cash_weight, self.index = 0.0, self.index + 1
        terminated = self.index >= len(self.prices) - 1
        info = {"net_return": net, "gross_return": gross, "turnover": turnover, "trading_cost": cost,
                "equity": self.equity, "drawdown": drawdown, "volatility": volatility,
                "benchmark_net_return": benchmark_net, "reward": float(reward),
                "legacy_turnover_penalty": turnover_term, "legacy_volatility_penalty": volatility_term,
                "legacy_drawdown_penalty": drawdown_term,
                "decision_date": decision_date, "return_date": str(self.dates[self.index])[:10]}
        return self._observation(), float(reward), terminated, False, info

    def _target_weights(self, action):
        scores = np.asarray(action)
        if scores.shape != (self.n_assets,):
            raise ValueError("Action must have one score per trained asset")
        return allocation_weights(scores, self.reward_config.max_weight,
                                  getattr(self, 'weights', None), self.reward_config.rebalance_rate)

    def _observation(self):
        prices = self.prices[self.index - self.window:self.index + 1]; returns = prices[1:] / prices[:-1] - 1
        log_returns = np.log(prices[1:] / prices[:-1]); rolling_vol = np.array([np.std(returns[max(0, t - 19):t + 1], axis=0) for t in range(len(returns))])
        ma_gap = prices[1:] / np.maximum(np.mean(prices[:-1], axis=0), 1e-6) - 1
        volume = self.volumes[self.index-self.window+1:self.index+1]; volume_z = (volume-volume.mean(0))/(volume.std(0)+1e-6)
        market = np.stack((returns, log_returns, rolling_vol, ma_gap, volume_z), axis=-1).astype(np.float32)
        risk = np.array([np.std(self.returns[-20:]) if len(self.returns)>1 else 0, 1-self.equity/self.peak_equity, self.equity/self.initial_cash-1], dtype=np.float32)
        observation = {"market": market, "portfolio": np.r_[self.weights, self.cash_weight].astype(np.float32), "risk": risk}
        if self.covariance_window:
            if self.index not in self._covariance_cache:
                from sklearn.covariance import LedoitWolf
                history=self.prices[self.index-self.covariance_window:self.index+1]
                if len(history)!=self.covariance_window+1:
                    raise ValueError('Insufficient known covariance history')
                known_returns=history[1:]/history[:-1]-1
                self._covariance_cache[self.index]=LedoitWolf().fit(known_returns).covariance_.astype(np.float32)
            observation['covariance']=self._covariance_cache[self.index].copy()
        return observation
