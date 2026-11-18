"""Historical paper account. No broker access; no model/preset action fallback."""
from pathlib import Path
import hashlib
import json
import threading
import uuid
import numpy as np
from finance.cash_execution import execute

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/finance_paper'
STRATEGIES = ['Equal weight', 'Inverse volatility', 'Momentum', 'Reversal']


class Panel:
    def __init__(self):
        self.manifest = json.loads((DATA/'manifest.json').read_text())
        if hashlib.sha256((DATA/'nifty_prices.npz').read_bytes()).hexdigest() != self.manifest['panel_sha256']:
            raise ValueError('Historical price panel integrity failure')
        with np.load(DATA/'nifty_prices.npz', allow_pickle=False) as z:
            self.prices, self.dates, self.tickers = z['prices'], z['dates'], z['tickers'].tolist()
        self.first = int(np.searchsorted(self.dates, np.datetime64(self.manifest['first_return']))) - 1
        self.last = int(np.searchsorted(self.dates, np.datetime64(self.manifest['last_return'])))
        assert self.first >= 253 and self.last < len(self.dates)
        self.cache = {}

    def targets(self, execution_index):
        # Execution close at index; the last permitted signal is index-1.
        i = execution_index - 1
        if i not in self.cache:
            p = self.prices
            r = p[i-125:i+1] / p[i-126:i] - 1
            vol = r.std(0).clip(1e-5)
            momentum = p[i-21]/p[i-126]-1
            short = p[i]/p[i-5]-1
            def tilt(x):
                z = (x-x.mean())/(x.std()+1e-8)
                w = np.exp(np.clip(z, -2, 2))
                return w/w.sum()
            inv = 1/vol
            inv /= inv.sum()
            t = np.stack([np.ones(len(vol))/len(vol), inv, tilt(momentum), tilt(-short)])
            self.cache[i] = .5*t + .5/len(vol)
        return self.cache[i].copy()


class PaperAccount:
    def __init__(self, panel, capital=100000.):
        self.panel = panel
        self.capital = self.equity = self.peak = float(capital)
        self.index = panel.first
        self.n = 0
        self.weights = np.zeros(len(panel.tickers))
        self.cash = 1.
        self.pnl = np.zeros(len(panel.tickers))
        self.fees = 0.
        self.history = []
        self.activities = []
        self.returns = []
        self.lock = threading.Lock()
        self.reference = {name: {'weights': np.zeros(len(panel.tickers)), 'cash': 1., 'equity': capital}
                          for name in ['untrained', 'buy_hold']}

    def step(self, predict):
        if self.index >= self.panel.last:
            return False
        p = self.panel
        rebalance = self.n % 5 == 0
        targets = p.targets(self.index) if rebalance else None
        receipt = None
        if rebalance:
            try:
                probs, receipt = predict(account=self)
            except (TypeError, ValueError):
                probs, receipt = predict()
            probs = np.asarray(probs, dtype=float)
            if probs.shape == (len(self.weights),):
                target = probs
                strategy_probs = [0.25] * 4
                contributions = np.zeros((4, len(self.weights)))
            elif probs.shape == (4,):
                if not np.isfinite(probs).all() or (probs < 0).any():
                    raise ValueError('Invalid real-model strategy probabilities')
                probs = probs/probs.sum()
                target = probs@targets
                strategy_probs = probs.tolist()
                contributions = probs[:, None]*targets
            else:
                raise ValueError(f'Invalid real-model action shape {probs.shape}')
        else:
            target = self.weights.copy()
        rel = p.prices[self.index+1]/p.prices[self.index]
        result = execute(self.weights, self.cash, target, rel)
        old_equity = self.equity
        if rebalance:
            traded = old_equity*(result.post_fee_stock_notional-self.weights)
            rationales = receipt.get('rationales', []) if isinstance(receipt, dict) else []
            self.activities.append({
                'id': len(self.activities), 'execution_date': str(p.dates[self.index]),
                'signal_date': str(p.dates[self.index-1]), 'probabilities': strategy_probs,
                'strategy_names': STRATEGIES, 'previous_weights': self.weights.tolist(),
                'target_weights': target.tolist(), 'strategy_contributions': contributions.tolist() if hasattr(contributions, 'tolist') else contributions,
                'pre_trade_equity': old_equity, 'fee': old_equity*result.trading_cost,
                'turnover': result.turnover, 'model_receipt': receipt, 'rationales': rationales,
                'trades': [{'ticker': t, 'amount': float(amount),
                            'previous_weight': float(self.weights[i]), 'target_weight': float(target[i])}
                           for i, (t, amount) in enumerate(zip(p.tickers, traded)) if abs(amount) > .005]})
        self.pnl += old_equity*result.post_fee_stock_notional*(rel-1)
        self.fees += old_equity*result.trading_cost
        self.weights, self.cash = result.stock_weights, result.cash_weight
        self.equity *= 1+result.net_return
        self.peak = max(self.peak, self.equity)
        self.returns.append(result.net_return)
        # Reference trajectories use recorded raw-model probabilities, not online inference.
        for name, ref in self.reference.items():
            if name == 'untrained' and rebalance:
                prob = np.asarray(p.manifest['recorded_probabilities']['untrained'], dtype=float)
                ref_target = (prob/prob.sum())@targets
            elif name == 'buy_hold' and self.n == 0:
                ref_target = np.ones(len(self.weights))/len(self.weights)
            else:
                ref_target = ref['weights'].copy()
            rr = execute(ref['weights'], ref['cash'], ref_target, rel)
            ref.update(weights=rr.stock_weights, cash=rr.cash_weight, equity=ref['equity']*(1+rr.net_return))
        self.index += 1
        self.n += 1
        self.history.append({'date': str(p.dates[self.index]), 'equity': self.equity,
                             'untrained': self.reference['untrained']['equity'],
                             'buy_hold': self.reference['buy_hold']['equity'],
                             'drawdown': 1-self.equity/self.peak})
        if abs(self.equity-self.capital-(self.pnl.sum()-self.fees)) > 1e-6:
            raise AssertionError('Market P&L minus fees does not reconcile to account gain')
        return True

    def snapshot(self):
        p = self.panel
        return {'date': str(p.dates[self.index]), 'initial_date': str(p.dates[p.first]),
                'capital': self.capital, 'equity': self.equity, 'cash': self.cash*self.equity,
                'net_return': self.equity/self.capital-1, 'fees': self.fees,
                'maximum_drawdown': max([h['drawdown'] for h in self.history], default=0.),
                'sharpe': float(np.mean(self.returns)/np.std(self.returns)*np.sqrt(252))
                if len(self.returns)>1 and np.std(self.returns)>0 else None,
                'sessions': self.n, 'total_sessions': p.last-p.first, 'finished': self.index >= p.last,
                'next_rebalance_in': (5-self.n % 5) % 5,
                'holdings': [{'ticker': t, 'value': float(w*self.equity), 'weight': float(w),
                              'cumulative_market_pnl': float(self.pnl[i])}
                             for i, (t, w) in enumerate(zip(p.tickers, self.weights))],
                'history': list(self.history), 'activities': list(self.activities),
                'currency': 'INR', 'mode': 'historical_paper_trading',
                'model': p.manifest['adapter'], 'start': p.manifest['first_return'],
                'end': p.manifest['last_return']}


_panel = None
_sessions = {}
_lock = threading.Lock()


def panel():
    global _panel
    if _panel is None:
        _panel = Panel()
    return _panel


def create(capital=100000.):
    with _lock:
        if len(_sessions) >= 32:
            raise ValueError('Close an existing paper session before opening another')
        key = uuid.uuid4().hex
        _sessions[key] = PaperAccount(panel(), capital)
        return key, _sessions[key]


def account(key):
    if key not in _sessions:
        raise KeyError('Paper session not found; start a new session')
    return _sessions[key]


def close(key):
    with _lock:
        _sessions.pop(key, None)
