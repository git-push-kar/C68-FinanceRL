"""Live Real-Time Market Service and Persistent Portfolio Manager for Nifty 49 Equities.

Supports:
  1. Live market data fetching via yfinance (Yahoo Finance NSE tickers).
  2. Persistent JSON storage for the live forward portfolio starting from January 1, 2026.
  3. Continuous day-by-day progression: runs up to today (Nov 18, 2026), closes with market close,
     and seamlessly resumes tomorrow (Nov 19, 2026) and day after tomorrow on real-time data.
  4. Real-time inference with InternVL3-2B + LoRA RL adapter on live portfolio state.
  5. Generating natural-language investment rationales embedded in execution logs.
"""

import os
import sys
import json
import time
import datetime
import threading
from pathlib import Path
import numpy as np
import pandas as pd

DEMO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = DEMO_ROOT / 'data' / 'finance_paper'
STORE_PATH = DATA_DIR / 'live_portfolio_store.json'
CACHE_PATH = DATA_DIR / 'live_market_cache.npz'

from finance.cash_execution import execute

TICKERS_49 = [
    'ADANIENT.NS', 'ADANIPORTS.NS', 'APOLLOHOSP.NS', 'ASIANPAINT.NS', 'AXISBANK.NS',
    'BAJAJ-AUTO.NS', 'BAJAJFINSV.NS', 'BAJFINANCE.NS', 'BHARTIARTL.NS', 'BPCL.NS',
    'BRITANNIA.NS', 'CIPLA.NS', 'COALINDIA.NS', 'DIVISLAB.NS', 'DRREDDY.NS',
    'EICHERMOT.NS', 'GRASIM.NS', 'HCLTECH.NS', 'HDFCBANK.NS', 'HDFCLIFE.NS',
    'HEROMOTOCO.NS', 'HINDALCO.NS', 'HINDUNILVR.NS', 'ICICIBANK.NS', 'INDUSINDBK.NS',
    'INFY.NS', 'ITC.NS', 'JSWSTEEL.NS', 'KOTAKBANK.NS', 'LT.NS',
    'LTIM.NS', 'M&M.NS', 'MARUTI.NS', 'NESTLEIND.NS', 'NTPC.NS',
    'ONGC.NS', 'POWERGRID.NS', 'RELIANCE.NS', 'SBILIFE.NS', 'SBIN.NS',
    'SHREECEM.NS', 'SUNPHARMA.NS', 'TATACONSUM.NS', 'TATASTEEL.NS', 'TCS.NS',
    'TECHM.NS', 'TITAN.NS', 'ULTRACEMCO.NS', 'WIPRO.NS'
]

YF_TICKERS = [t.replace('LTIM.NS', 'LTM.NS') for t in TICKERS_49]


class LiveMarketService:
    """Fetches, caches, and provides current and real-time market data for all 49 Nifty equities."""

    @classmethod
    def fetch_latest_data(cls, start_date='2025-06-01'):
        """Download latest historical and current prices from Yahoo Finance."""
        try:
            import yfinance as yf
            end_date = (datetime.date.today() + datetime.timedelta(days=1)).strftime('%Y-%m-%d')
            df = yf.download(YF_TICKERS, start=start_date, end=end_date, group_by='ticker', progress=False)
            if df.empty:
                return None
            
            dates = [d.strftime('%Y-%m-%d') for d in df.index]
            n_dates = len(dates)
            n_tickers = len(TICKERS_49)

            open_arr = np.zeros((n_dates, n_tickers), dtype=np.float64)
            close_arr = np.zeros((n_dates, n_tickers), dtype=np.float64)
            adj_close_arr = np.zeros((n_dates, n_tickers), dtype=np.float64)
            vol_arr = np.zeros((n_dates, n_tickers), dtype=np.float64)

            for i, (t, yt) in enumerate(zip(TICKERS_49, YF_TICKERS)):
                open_arr[:, i] = df[yt]['Open'].to_numpy()
                close_arr[:, i] = df[yt]['Close'].to_numpy()
                adj_close_arr[:, i] = df[yt]['Close'].to_numpy()
                vol_arr[:, i] = df[yt]['Volume'].to_numpy()

            DATA_DIR.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                CACHE_PATH,
                dates=np.array(dates),
                tickers=np.array(TICKERS_49),
                open=open_arr,
                close=close_arr,
                prices=adj_close_arr,
                volume=vol_arr
            )
            return CACHE_PATH
        except Exception as e:
            print(f"[LiveMarketService] Warning: Could not fetch online quotes ({e}). Using local cache.")
            return CACHE_PATH if CACHE_PATH.exists() else None

    @classmethod
    def load_panel(cls):
        """Load the cached live market panel."""
        if not CACHE_PATH.exists():
            cls.fetch_latest_data()
        if not CACHE_PATH.exists():
            raise FileNotFoundError(f"Market cache not found at {CACHE_PATH}")
        data = np.load(CACHE_PATH, allow_pickle=True)
        return {
            'dates': list(data['dates']),
            'tickers': list(data['tickers']),
            'prices': data['prices'],
            'open': data['open'],
            'close': data['close'],
            'volume': data['volume']
        }


class LivePortfolioManager:
    """Persistent paper portfolio manager operating on real-time market data from Jan 1, 2026 onwards."""

    def __init__(self, capital=100000.0):
        self.lock = threading.Lock()
        self.capital = float(capital)
        self.panel_data = LiveMarketService.load_panel()
        self.dates = self.panel_data['dates']
        self.tickers = self.panel_data['tickers']
        self.prices = self.panel_data['prices']
        
        # Locate Jan 1, 2026
        self.start_index = next(i for i, d in enumerate(self.dates) if d >= '2026-01-01')
        self.state = None
        self._load_or_bootstrap()

    @property
    def panel(self):
        """Duck-typed panel interface matching PaperAccount requirements for predict()."""
        class _PanelAdapter:
            def __init__(self, mgr):
                self.prices = mgr.prices
                self.dates = mgr.dates
                self.tickers = mgr.tickers
        return _PanelAdapter(self)

    @property
    def index(self):
        return self.state['current_index']

    @property
    def cash(self):
        return self.state['cash_weight']

    @property
    def equity(self):
        return self.state['equity']

    @property
    def weights(self):
        return np.array(self.state['weights'], dtype=np.float64)

    def _load_or_bootstrap(self):
        """Load state from persistent JSON store or run bootstrap simulation from Jan 1, 2026."""
        if STORE_PATH.exists():
            try:
                with open(STORE_PATH, 'r', encoding='utf-8') as f:
                    self.state = json.load(f)
                # Verify date is within panel
                if self.state.get('capital') == self.capital and self.state.get('current_index') < len(self.dates):
                    return
            except Exception as e:
                print(f"[LivePortfolioManager] Failed to load existing store: {e}. Rebuilding...")

        self._bootstrap_from_jan_2026()

    def _bootstrap_from_jan_2026(self):
        """Bootstrap the model-managed portfolio from Jan 1, 2026 to the latest available session."""
        print(f"[LivePortfolioManager] Bootstrapping live portfolio from Jan 1, 2026 (index {self.start_index}) to index {len(self.dates)-1}...")
        from finance.policy import predict

        capital = float(self.capital)
        equity = peak = capital
        n_stocks = len(self.tickers)
        weights = np.zeros(n_stocks, dtype=np.float64)
        cash_weight = 1.0
        pnl = np.zeros(n_stocks, dtype=np.float64)
        fees = 0.0
        history = []
        activities = []
        returns = []

        # Reference trajectories
        ref_untrained = {'weights': np.zeros(n_stocks), 'cash': 1.0, 'equity': capital}
        ref_buy_hold = {'weights': np.zeros(n_stocks), 'cash': 1.0, 'equity': capital}

        last_index = len(self.dates) - 1

        for idx in range(self.start_index, last_index):
            session_num = idx - self.start_index
            rebalance = (session_num % 5 == 0)

            # Duck-typed account object for policy.predict()
            class _Acc:
                pass
            acc = _Acc()
            acc.panel = self.panel
            acc.index = idx
            acc.cash = cash_weight
            acc.equity = equity
            acc.weights = weights

            if rebalance:
                try:
                    target_w, receipt = predict(account=acc)
                except Exception as e:
                    print(f"[LivePortfolioManager] Predict fallback at index {idx}: {e}")
                    target_w = np.ones(n_stocks) / n_stocks
                    receipt = {'mode': 'fallback', 'adapter': 'fallback'}

                old_equity = equity
                rel = self.prices[idx + 1] / np.maximum(self.prices[idx], 1e-4)
                res = execute(weights, cash_weight, target_w, rel)

                traded = old_equity * (res.post_fee_stock_notional - weights)
                rationales = receipt.get('rationales', []) if isinstance(receipt, dict) else []
                activities.append({
                    'id': len(activities),
                    'execution_date': self.dates[idx],
                    'signal_date': self.dates[idx - 1],
                    'probabilities': [0.25] * 4,
                    'strategy_names': ['Top-15 Conviction Pick', 'Momentum Quality', 'Low Volatility', 'Cash Reserve'],
                    'previous_weights': weights.tolist(),
                    'target_weights': target_w.tolist(),
                    'strategy_contributions': np.zeros((4, n_stocks)).tolist(),
                    'pre_trade_equity': float(old_equity),
                    'fee': float(old_equity * res.trading_cost),
                    'turnover': float(res.turnover),
                    'model_receipt': receipt,
                    'rationales': rationales,
                    'trades': [{'ticker': t, 'amount': float(amount),
                                'previous_weight': float(weights[i]), 'target_weight': float(target_w[i])}
                               for i, (t, amount) in enumerate(zip(self.tickers, traded)) if abs(amount) > 0.005]
                })

                pnl += old_equity * res.post_fee_stock_notional * (rel - 1.0)
                fees += old_equity * res.trading_cost
                weights, cash_weight = res.stock_weights, res.cash_weight
                equity *= (1.0 + res.net_return)
            else:
                old_equity = equity
                rel = self.prices[idx + 1] / np.maximum(self.prices[idx], 1e-4)
                res = execute(weights, cash_weight, weights, rel)
                pnl += old_equity * res.post_fee_stock_notional * (rel - 1.0)
                weights, cash_weight = res.stock_weights, res.cash_weight
                equity *= (1.0 + res.net_return)

            peak = max(peak, equity)
            returns.append(res.net_return)

            # Update Untrained Reference
            if rebalance:
                ref_un_target = np.ones(n_stocks) / n_stocks
            else:
                ref_un_target = ref_untrained['weights'].copy()
            res_un = execute(ref_untrained['weights'], ref_untrained['cash'], ref_un_target, rel)
            ref_untrained.update(weights=res_un.stock_weights, cash=res_un.cash_weight, equity=ref_untrained['equity'] * (1.0 + res_un.net_return))

            # Update Buy & Hold Reference
            if session_num == 0:
                ref_bh_target = np.ones(n_stocks) / n_stocks
            else:
                ref_bh_target = ref_buy_hold['weights'].copy()
            res_bh = execute(ref_buy_hold['weights'], ref_buy_hold['cash'], ref_bh_target, rel)
            ref_buy_hold.update(weights=res_bh.stock_weights, cash=res_bh.cash_weight, equity=ref_buy_hold['equity'] * (1.0 + res_bh.net_return))

            history.append({
                'date': self.dates[idx + 1],
                'equity': float(equity),
                'untrained': float(ref_untrained['equity']),
                'buy_hold': float(ref_buy_hold['equity']),
                'drawdown': float(1.0 - equity / peak)
            })

        self.state = {
            'capital': capital,
            'current_index': last_index,
            'start_index': self.start_index,
            'asof_date': self.dates[last_index],
            'initial_date': self.dates[self.start_index],
            'equity': float(equity),
            'cash_weight': float(cash_weight),
            'weights': weights.tolist(),
            'pnl': pnl.tolist(),
            'fees': float(fees),
            'peak': float(peak),
            'returns': [float(r) for r in returns],
            'history': history,
            'activities': activities,
            'ref_untrained': {'weights': ref_untrained['weights'].tolist(), 'cash': float(ref_untrained['cash']), 'equity': float(ref_untrained['equity'])},
            'ref_buy_hold': {'weights': ref_buy_hold['weights'].tolist(), 'cash': float(ref_buy_hold['cash']), 'equity': float(ref_buy_hold['equity'])},
            'sessions_count': len(history),
            'updated_at': datetime.datetime.now().isoformat()
        }
        self.save()
        print(f"[LivePortfolioManager] Bootstrap complete! As of {self.dates[last_index]}: Equity=Rs {equity:.2f}, Return={(equity/capital-1)*100:+.2f}%, Fees=Rs {fees:.2f}")

    def save(self):
        """Persist state to JSON store."""
        with open(STORE_PATH, 'w', encoding='utf-8') as f:
            json.dump(self.state, f, indent=2)

    def advance_live_day(self):
        """Advance portfolio with newly fetched real-time market data for tomorrow."""
        with self.lock:
            # 1. Check for newer dates from live service
            LiveMarketService.fetch_latest_data()
            self.panel_data = LiveMarketService.load_panel()
            self.dates = self.panel_data['dates']
            self.prices = self.panel_data['prices']

            curr_idx = self.state['current_index']
            if curr_idx >= len(self.dates) - 1:
                # Up to date with latest available market session
                return self.snapshot()

            idx = curr_idx
            session_num = idx - self.start_index
            rebalance = (session_num % 5 == 0)

            from finance.policy import predict
            capital = self.state['capital']
            equity = self.state['equity']
            peak = self.state['peak']
            weights = np.array(self.state['weights'], dtype=np.float64)
            cash_weight = self.state['cash_weight']
            pnl = np.array(self.state['pnl'], dtype=np.float64)
            fees = self.state['fees']
            activities = self.state['activities']
            history = self.state['history']
            returns = self.state['returns']

            ref_untrained = {k: np.array(v) if isinstance(v, list) else v for k, v in self.state['ref_untrained'].items()}
            ref_buy_hold = {k: np.array(v) if isinstance(v, list) else v for k, v in self.state['ref_buy_hold'].items()}

            class _Acc:
                pass
            acc = _Acc()
            acc.panel = self.panel
            acc.index = idx
            acc.cash = cash_weight
            acc.equity = equity
            acc.weights = weights

            rel = self.prices[idx + 1] / np.maximum(self.prices[idx], 1e-4)

            if rebalance:
                target_w, receipt = predict(account=acc)
                old_equity = equity
                res = execute(weights, cash_weight, target_w, rel)

                traded = old_equity * (res.post_fee_stock_notional - weights)
                rationales = receipt.get('rationales', []) if isinstance(receipt, dict) else []
                activities.append({
                    'id': len(activities),
                    'execution_date': self.dates[idx],
                    'signal_date': self.dates[idx - 1],
                    'probabilities': [0.25] * 4,
                    'strategy_names': ['Top-15 Conviction Pick', 'Momentum Quality', 'Low Volatility', 'Cash Reserve'],
                    'previous_weights': weights.tolist(),
                    'target_weights': target_w.tolist(),
                    'strategy_contributions': np.zeros((4, len(weights))).tolist(),
                    'pre_trade_equity': float(old_equity),
                    'fee': float(old_equity * res.trading_cost),
                    'turnover': float(res.turnover),
                    'model_receipt': receipt,
                    'rationales': rationales,
                    'trades': [{'ticker': t, 'amount': float(amount),
                                'previous_weight': float(weights[i]), 'target_weight': float(target_w[i])}
                               for i, (t, amount) in enumerate(zip(self.tickers, traded)) if abs(amount) > 0.005]
                })

                pnl += old_equity * res.post_fee_stock_notional * (rel - 1.0)
                fees += old_equity * res.trading_cost
                weights, cash_weight = res.stock_weights, res.cash_weight
                equity *= (1.0 + res.net_return)
            else:
                old_equity = equity
                res = execute(weights, cash_weight, weights, rel)
                pnl += old_equity * res.post_fee_stock_notional * (rel - 1.0)
                weights, cash_weight = res.stock_weights, res.cash_weight
                equity *= (1.0 + res.net_return)

            peak = max(peak, equity)
            returns.append(res.net_return)

            # References
            res_un = execute(ref_untrained['weights'], ref_untrained['cash'], ref_untrained['weights'], rel)
            ref_untrained.update(weights=res_un.stock_weights, cash=res_un.cash_weight, equity=ref_untrained['equity'] * (1.0 + res_un.net_return))

            res_bh = execute(ref_buy_hold['weights'], ref_buy_hold['cash'], ref_buy_hold['weights'], rel)
            ref_buy_hold.update(weights=res_bh.stock_weights, cash=res_bh.cash_weight, equity=ref_buy_hold['equity'] * (1.0 + res_bh.net_return))

            history.append({
                'date': self.dates[idx + 1],
                'equity': float(equity),
                'untrained': float(ref_untrained['equity']),
                'buy_hold': float(ref_buy_hold['equity']),
                'drawdown': float(1.0 - equity / peak)
            })

            self.state['current_index'] = idx + 1
            self.state['asof_date'] = self.dates[idx + 1]
            self.state['equity'] = float(equity)
            self.state['cash_weight'] = float(cash_weight)
            self.state['weights'] = weights.tolist()
            self.state['pnl'] = pnl.tolist()
            self.state['fees'] = float(fees)
            self.state['peak'] = float(peak)
            self.state['returns'] = [float(r) for r in returns]
            self.state['history'] = history
            self.state['activities'] = activities
            self.state['sessions_count'] = len(history)
            self.state['ref_untrained'] = {'weights': ref_untrained['weights'].tolist(), 'cash': float(ref_untrained['cash']), 'equity': float(ref_untrained['equity'])}
            self.state['ref_buy_hold'] = {'weights': ref_buy_hold['weights'].tolist(), 'cash': float(ref_buy_hold['cash']), 'equity': float(ref_buy_hold['equity'])}
            self.state['updated_at'] = datetime.datetime.now().isoformat()
            self.save()
            return self.snapshot()

    def snapshot(self):
        """Format full portfolio state for UI consumption."""
        s = self.state
        cap = s['capital']
        eq = s['equity']
        weights = np.array(s['weights'], dtype=np.float64)
        pnl = np.array(s['pnl'], dtype=np.float64)
        cash_val = s['cash_weight'] * eq

        rets = s['returns']
        sharpe = float(np.mean(rets) / np.std(rets) * np.sqrt(252)) if len(rets) > 1 and np.std(rets) > 0 else 0.0
        drawdowns = [h['drawdown'] for h in s['history']]
        max_dd = max(drawdowns) if drawdowns else 0.0

        n_sessions = s['sessions_count']
        next_reb = (5 - n_sessions % 5) % 5

        holdings = []
        for i, (t, w) in enumerate(zip(self.tickers, weights)):
            holdings.append({
                'ticker': t,
                'value': float(w * eq),
                'weight': float(w),
                'cumulative_market_pnl': float(pnl[i])
            })

        return {
            'session_id': 'live_market_session',
            'mode': 'live',
            'is_live': True,
            'date': s['asof_date'],
            'initial_date': s['initial_date'],
            'capital': cap,
            'equity': eq,
            'cash': cash_val,
            'net_return': eq / cap - 1.0,
            'fees': s['fees'],
            'maximum_drawdown': max_dd,
            'sharpe': sharpe,
            'sessions': n_sessions,
            'total_sessions': n_sessions,
            'finished': False, # Always active and forward-looking
            'next_rebalance_in': next_reb,
            'holdings': holdings,
            'history': s['history'],
            'activities': s['activities'],
            'currency': 'INR',
            'model': 'InternVL3-2B-State-RL',
            'market_status': 'Live Market Tracking · Resumes automatically on next trading session',
            'updated_at': s['updated_at']
        }


_live_manager = None
_live_lock = threading.Lock()

def get_live_manager(capital=100000.0):
    global _live_manager
    with _live_lock:
        if _live_manager is None or _live_manager.capital != capital:
            _live_manager = LivePortfolioManager(capital)
        return _live_manager
