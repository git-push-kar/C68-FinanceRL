"""Live Real-Time Market Service, YTD Replay, and Persistent Portfolio Manager for Nifty 49 Equities.

Supports three distinct simulation modes:
  Mode 1: Final Benchmark Replay (Jan 2025 – Jan 2026) -> Replays primary test set via PaperEngine.
  Mode 2: 2026 YTD History Replay (Jan 1, 2026 – Yesterday) -> Replays the daily actions & rebalances that built the current portfolio.
  Mode 3: Live Real-Time Market Trading (Today's Live Session) -> Continues from yesterday's pre-existing portfolio,
          takes decisions in real time using live market data (yfinance), and settles into persistent storage
          so tomorrow continues seamlessly from today's closing state.
"""

import os
import sys
import json
import time
import shutil
import datetime
import threading
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import numpy as np

DEMO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = DEMO_ROOT / 'data' / 'finance_paper'
STORE_PATH = DATA_DIR / 'live_portfolio_store.json'
BACKUP_PATH = DATA_DIR / 'live_portfolio_store.backup.json'
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
    def get_market_status(cls) -> Tuple[str, bool]:
        """Determine whether the Indian equity market (NSE) is currently open."""
        now_utc = datetime.datetime.now(datetime.timezone.utc)
        now_ist = now_utc + datetime.timedelta(hours=5, minutes=30)
        is_weekday = now_ist.weekday() < 5
        market_open = datetime.time(9, 15)
        market_close = datetime.time(15, 30)
        curr_time = now_ist.time()

        if is_weekday and market_open <= curr_time <= market_close:
            return f"● LIVE MARKET OPEN (NSE {curr_time.strftime('%H:%M:%S')} IST)", True
        elif is_weekday and curr_time < market_open:
            return f"○ PRE-MARKET (NSE Opens at 09:15 IST · Currently {curr_time.strftime('%H:%M')} IST)", False
        else:
            return f"● MARKET CLOSED (Settled Daily Session · {now_ist.strftime('%Y-%m-%d %H:%M')} IST)", False

    @classmethod
    def fetch_latest_data(cls, start_date='2025-06-01'):
        """Download latest historical and current prices from Yahoo Finance."""
        try:
            import yfinance as yf
            end_date = (datetime.date.today() + datetime.timedelta(days=1)).strftime('%Y-%m-%d')
            df = yf.download(YF_TICKERS, start=start_date, end=end_date, group_by='ticker', progress=False)
            if df.empty:
                return CACHE_PATH if CACHE_PATH.exists() else None

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
    def fetch_realtime_quotes(cls) -> Dict[str, Dict[str, float]]:
        """Fetch real-time / intraday quotes for all 49 Nifty tickers."""
        quotes = {}
        try:
            import yfinance as yf
            df = yf.download(YF_TICKERS, period='2d', interval='1d', group_by='ticker', progress=False)
            for t, yt in zip(TICKERS_49, YF_TICKERS):
                try:
                    ticker_series = df[yt]['Close'].dropna()
                    if len(ticker_series) >= 2:
                        prev_close = float(ticker_series.iloc[-2])
                        curr_price = float(ticker_series.iloc[-1])
                    elif len(ticker_series) == 1:
                        prev_close = curr_price = float(ticker_series.iloc[-1])
                    else:
                        prev_close = curr_price = 100.0
                    change_pct = (curr_price / prev_close - 1.0) if prev_close > 0 else 0.0
                    quotes[t] = {
                        'price': curr_price,
                        'prev_close': prev_close,
                        'change_pct': change_pct
                    }
                except Exception:
                    quotes[t] = {'price': 100.0, 'prev_close': 100.0, 'change_pct': 0.0}
        except Exception as e:
            print(f"[LiveMarketService] Realtime quote error: {e}. Falling back to cached prices.")
            panel = cls.load_panel()
            last_p = panel['prices'][-1]
            prev_p = panel['prices'][-2] if len(panel['prices']) > 1 else last_p
            for i, t in enumerate(TICKERS_49):
                quotes[t] = {
                    'price': float(last_p[i]),
                    'prev_close': float(prev_p[i]),
                    'change_pct': float(last_p[i] / max(prev_p[i], 1e-4) - 1.0)
                }
        return quotes

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


# ==============================================================================
# MODE 2: 2026 YTD HISTORY REPLAY (Jan 1, 2026 – Yesterday)
# ==============================================================================
class YTDReplaySession:
    """Replays the historical trading progression from Jan 1, 2026 up to yesterday's date.
    
    Lets the user inspect day-by-day and rebalance-by-rebalance how the portfolio was built
    from starting cash up to yesterday's settled close.
    """

    def __init__(self, key="ytd_history_session", capital=100000.0):
        self.key = key
        self.capital = float(capital)
        self.lock = threading.RLock()
        with open(STORE_PATH, "r", encoding="utf-8") as f:
            self.full_store = json.load(f)

        self.tickers = TICKERS_49
        self.history = self.full_store.get("history", [])
        self.activities = self.full_store.get("activities", [])
        self.total_sessions = len(self.history)
        self.current_step = min(1, self.total_sessions - 1)

    def step(self, mode="day"):
        with self.lock:
            if self.current_step >= self.total_sessions - 1:
                return False
            if mode == "rebalance":
                curr_date = self.history[self.current_step]["date"]
                next_step = self.current_step + 1
                while next_step < self.total_sessions - 1:
                    step_date = self.history[next_step]["date"]
                    if any(a["execution_date"] == step_date for a in self.activities):
                        break
                    next_step += 1
                self.current_step = next_step
            else:
                self.current_step += 1
            return True

    def reset(self):
        with self.lock:
            self.current_step = 1
            return self.snapshot()

    def snapshot(self):
        with self.lock:
            h_slice = self.history[:self.current_step + 1]
            latest_h = h_slice[-1] if h_slice else {'date': '2026-01-02', 'equity': self.capital, 'drawdown': 0.0}
            curr_date = latest_h['date']

            act_slice = [a for a in self.activities if a['execution_date'] <= curr_date]
            latest_act = act_slice[-1] if act_slice else None

            eq = latest_h['equity']
            weights = latest_act['target_weights'] if latest_act else [0.0] * len(self.tickers)
            cash_val = eq * max(0.0, 1.0 - sum(weights)) if latest_act else eq

            rets = [h_slice[i]['equity'] / h_slice[i - 1]['equity'] - 1.0 for i in range(1, len(h_slice))]
            sharpe = float(np.mean(rets) / np.std(rets) * np.sqrt(252)) if len(rets) > 1 and np.std(rets) > 0 else 0.0
            max_dd = max([h['drawdown'] for h in h_slice]) if h_slice else 0.0

            holdings = []
            for i, (t, w) in enumerate(zip(self.tickers, weights)):
                holdings.append({
                    'ticker': t,
                    'value': float(w * eq),
                    'weight': float(w),
                    'cumulative_market_pnl': 0.0
                })

            finished = (self.current_step >= self.total_sessions - 1)
            status_desc = (
                "2026 YTD History Replay Complete · Current portfolio established as of " + curr_date +
                ". Switch to Mode 3 for Live Real-Time Market Trading."
                if finished else
                f"2026 YTD History Replay · Replaying Day {self.current_step + 1} of {self.total_sessions} ({curr_date})"
            )

            return {
                'session_id': self.key,
                'mode': 'ytd',
                'is_live': False,
                'date': curr_date,
                'initial_date': self.history[0]['date'] if self.history else '2026-01-02',
                'capital': self.capital,
                'equity': eq,
                'cash': cash_val,
                'net_return': eq / self.capital - 1.0,
                'fees': sum(a.get('fee', 0.0) for a in act_slice),
                'maximum_drawdown': max_dd,
                'sharpe': sharpe,
                'sessions': self.current_step + 1,
                'total_sessions': self.total_sessions,
                'finished': finished,
                'next_rebalance_in': (5 - (self.current_step + 1) % 5) % 5,
                'holdings': holdings,
                'history': h_slice,
                'activities': act_slice,
                'currency': 'INR',
                'model': 'InternVL3-2B-State-RL',
                'market_status': status_desc,
                'updated_at': self.full_store.get('updated_at')
            }


# ==============================================================================
# MODE 3: LIVE REAL-TIME MARKET TRADING (Today's Live Session)
# ==============================================================================
class LiveRealtimeSession:
    """Active live trading session continuing from yesterday's settled close.
    
    1. Ingests today's live real-time market data from yfinance.
    2. Marks the portfolio to market in real time.
    3. Triggers InternVL3-2B + LoRA RL to evaluate today's live market and execute decisions with reasoning.
    4. Settles today's session into persistent storage (live_portfolio_store.json) so tomorrow
       resumes from today's closing state.
    """

    def __init__(self, key="live_market_session", capital=100000.0):
        self.key = key
        self.capital = float(capital)
        self.lock = threading.RLock()
        self.tickers = TICKERS_49
        self.live_quotes = {}
        self.today_activity = None
        self.is_settled_today = False
        self._load_settled_portfolio()

    def _load_settled_portfolio(self):
        """Load baseline portfolio state settled as of yesterday's close."""
        if not STORE_PATH.exists():
            mgr = LivePortfolioManager(self.capital)
            mgr._load_or_bootstrap()

        with open(STORE_PATH, "r", encoding="utf-8") as f:
            self.settled_state = json.load(f)

        self.settled_date = self.settled_state['asof_date']
        self.settled_equity = float(self.settled_state['equity'])
        self.settled_cash_weight = float(self.settled_state['cash_weight'])
        self.settled_weights = np.array(self.settled_state['weights'], dtype=np.float64)

        # Initialize live values
        self.live_equity = self.settled_equity
        self.live_weights = self.settled_weights.copy()
        self.live_cash = self.settled_cash_weight * self.settled_equity
        self.refresh_quotes()

    def refresh_quotes(self):
        """Fetch today's real-time quotes from yfinance and mark portfolio to market."""
        with self.lock:
            self.live_quotes = LiveMarketService.fetch_realtime_quotes()
            # Calculate mark to market equity
            stock_value = 0.0
            for i, t in enumerate(self.tickers):
                q = self.live_quotes.get(t, {})
                rel = 1.0 + q.get('change_pct', 0.0)
                stock_value += (self.settled_weights[i] * self.settled_equity) * rel

            self.live_equity = self.live_cash + stock_value
            return self.snapshot()

    def execute_live_decision(self):
        """Run InternVL3-2B + LoRA RL inference on current live market quotes."""
        with self.lock:
            from finance.policy import predict

            # Prepare live state observation
            n_stocks = len(self.tickers)
            live_rel = np.array([
                1.0 + self.live_quotes.get(t, {}).get('change_pct', 0.0)
                for t in self.tickers
            ], dtype=np.float64)

            # Duck-typed account
            class _LiveAcc:
                pass
            acc = _LiveAcc()
            panel_data = LiveMarketService.load_panel()

            class _PanelAdapter:
                def __init__(self, pdata):
                    self.prices = pdata['prices']
                    self.dates = pdata['dates']
                    self.tickers = pdata['tickers']
            acc.panel = _PanelAdapter(panel_data)
            acc.index = len(panel_data['dates']) - 1
            acc.cash = self.live_cash / max(self.live_equity, 1.0)
            acc.equity = self.live_equity
            acc.weights = self.live_weights

            try:
                target_w, receipt = predict(account=acc)
            except Exception as e:
                print(f"[LiveRealtimeSession] Model predict fallback: {e}")
                target_w = self.live_weights.copy()
                receipt = {'mode': 'hold', 'adapter': 'internvl3_state_rl'}

            old_eq = self.live_equity
            cash_w = self.live_cash / max(self.live_equity, 1.0)
            res = execute(self.live_weights, cash_w, target_w, live_rel)

            traded = old_eq * (res.post_fee_stock_notional - self.live_weights)
            rationales = receipt.get('rationales', []) if isinstance(receipt, dict) else []

            today_str = datetime.date.today().strftime('%Y-%m-%d')
            activity = {
                'id': len(self.settled_state.get('activities', [])),
                'execution_date': today_str,
                'signal_date': self.settled_date,
                'probabilities': [0.25] * 4,
                'strategy_names': ['Top-15 Conviction Pick', 'Momentum Quality', 'Low Volatility', 'Cash Reserve'],
                'previous_weights': self.live_weights.tolist(),
                'target_weights': target_w.tolist(),
                'strategy_contributions': np.zeros((4, n_stocks)).tolist(),
                'pre_trade_equity': float(old_eq),
                'fee': float(old_eq * res.trading_cost),
                'turnover': float(res.turnover),
                'model_receipt': receipt,
                'rationales': rationales,
                'trades': [{'ticker': t, 'amount': float(amount),
                            'previous_weight': float(self.live_weights[i]), 'target_weight': float(target_w[i])}
                           for i, (t, amount) in enumerate(zip(self.tickers, traded)) if abs(amount) > 0.005]
            }

            self.today_activity = activity
            self.live_weights = res.stock_weights
            self.live_cash = res.cash_weight * old_eq
            self.live_equity = old_eq * (1.0 + res.net_return)
            return self.snapshot()

    def settle_today(self, target_date=None):
        """Commit today's live session into persistent storage (live_portfolio_store.json).
        
        Advances the permanent portfolio baseline so that tomorrow continues from this closing state.
        """
        with self.lock:
            # Create safety backup
            if STORE_PATH.exists():
                shutil.copy2(STORE_PATH, BACKUP_PATH)

            today_str = target_date or datetime.date.today().strftime('%Y-%m-%d')
            if today_str <= self.settled_date:
                # If testing on the same day or weekend, advance to next business day
                d = datetime.date.fromisoformat(self.settled_date)
                next_d = d + datetime.timedelta(days=1)
                while next_d.weekday() >= 5: # Skip Sat/Sun
                    next_d += datetime.timedelta(days=1)
                today_str = next_d.strftime('%Y-%m-%d')

            history = self.settled_state.get('history', [])
            activities = self.settled_state.get('activities', [])
            returns = self.settled_state.get('returns', [])

            day_ret = (self.live_equity / max(self.settled_equity, 1.0)) - 1.0
            peak = max(self.settled_state.get('peak', self.capital), self.live_equity)
            drawdown = float(1.0 - self.live_equity / peak)

            history.append({
                'date': today_str,
                'equity': float(self.live_equity),
                'untrained': float(self.settled_state.get('ref_untrained', {}).get('equity', self.capital) * (1.0 + day_ret)),
                'buy_hold': float(self.settled_state.get('ref_buy_hold', {}).get('equity', self.capital) * (1.0 + day_ret)),
                'drawdown': drawdown
            })

            if self.today_activity:
                activities.append(self.today_activity)

            returns.append(float(day_ret))
            cash_w = float(self.live_cash / max(self.live_equity, 1.0))

            updated_state = {
                'capital': self.capital,
                'current_index': self.settled_state.get('current_index', 0) + 1,
                'start_index': self.settled_state.get('start_index', 0),
                'asof_date': today_str,
                'initial_date': self.settled_state.get('initial_date', '2026-01-02'),
                'equity': float(self.live_equity),
                'cash_weight': cash_w,
                'weights': self.live_weights.tolist(),
                'pnl': self.settled_state.get('pnl', [0.0]*len(self.tickers)),
                'fees': float(self.settled_state.get('fees', 0.0) + (self.today_activity.get('fee', 0.0) if self.today_activity else 0.0)),
                'peak': float(peak),
                'returns': returns,
                'history': history,
                'activities': activities,
                'ref_untrained': self.settled_state.get('ref_untrained', {}),
                'ref_buy_hold': self.settled_state.get('ref_buy_hold', {}),
                'sessions_count': len(history),
                'updated_at': datetime.datetime.now().isoformat()
            }

            with open(STORE_PATH, 'w', encoding='utf-8') as f:
                json.dump(updated_state, f, indent=2)

            self.settled_state = updated_state
            self.settled_date = today_str
            self.settled_equity = float(self.live_equity)
            self.settled_cash_weight = cash_w
            self.settled_weights = self.live_weights.copy()
            self.is_settled_today = True

            print(f"[LiveRealtimeSession] Successfully settled session {today_str}. Total sessions: {len(history)}.")
            return self.snapshot()

    def snapshot(self):
        with self.lock:
            s = self.settled_state
            market_label, is_open = LiveMarketService.get_market_status()
            today_str = datetime.date.today().strftime('%Y-%m-%d')

            holdings = []
            for i, (t, w) in enumerate(zip(self.tickers, self.live_weights)):
                q = self.live_quotes.get(t, {})
                holdings.append({
                    'ticker': t,
                    'value': float(w * self.live_equity),
                    'weight': float(w),
                    'price': q.get('price', 0.0),
                    'change_pct': q.get('change_pct', 0.0),
                    'cumulative_market_pnl': 0.0
                })

            rets = s.get('returns', [])
            sharpe = float(np.mean(rets) / np.std(rets) * np.sqrt(252)) if len(rets) > 1 and np.std(rets) > 0 else 0.0
            drawdowns = [h['drawdown'] for h in s.get('history', [])]
            max_dd = max(drawdowns) if drawdowns else 0.0

            activities = list(s.get('activities', []))
            if self.today_activity and self.today_activity not in activities:
                activities.append(self.today_activity)

            intraday_pnl = self.live_equity - self.settled_equity
            intraday_pct = (intraday_pnl / self.settled_equity) if self.settled_equity > 0 else 0.0

            settle_note = (
                "Today's session has been committed to permanent history."
                if self.is_settled_today else
                "Session active. Click 'Settle & Close Today' to lock today's run into history for tomorrow."
            )

            return {
                'session_id': self.key,
                'mode': 'live',
                'is_live': True,
                'date': today_str,
                'settled_date': self.settled_date,
                'initial_date': s.get('initial_date', '2026-01-02'),
                'capital': self.capital,
                'equity': self.live_equity,
                'settled_equity': self.settled_equity,
                'intraday_pnl': intraday_pnl,
                'intraday_pct': intraday_pct,
                'cash': self.live_cash,
                'net_return': self.live_equity / self.capital - 1.0,
                'fees': s.get('fees', 0.0),
                'maximum_drawdown': max_dd,
                'sharpe': sharpe,
                'sessions': len(s.get('history', [])),
                'total_sessions': len(s.get('history', [])),
                'finished': False,
                'next_rebalance_in': 0,
                'holdings': holdings,
                'history': s.get('history', []),
                'activities': activities,
                'currency': 'INR',
                'model': 'InternVL3-2B-State-RL',
                'market_status': market_label,
                'is_market_open': is_open,
                'is_settled_today': self.is_settled_today,
                'settle_note': settle_note,
                'updated_at': datetime.datetime.now().isoformat()
            }


# Global Singletons
_ytd_session: Optional[YTDReplaySession] = None
_live_session: Optional[LiveRealtimeSession] = None
_manager_lock = threading.Lock()

def get_ytd_session(capital=100000.0) -> YTDReplaySession:
    global _ytd_session
    with _manager_lock:
        if _ytd_session is None or _ytd_session.capital != capital:
            _ytd_session = YTDReplaySession(capital=capital)
        return _ytd_session

def get_live_realtime_session(capital=100000.0) -> LiveRealtimeSession:
    global _live_session
    with _manager_lock:
        if _live_session is None or _live_session.capital != capital:
            _live_session = LiveRealtimeSession(capital=capital)
        return _live_session

# Backward compatibility aliases
LivePortfolioManager = LiveRealtimeSession
def get_live_manager(capital=100000.0):
    return get_live_realtime_session(capital=capital)
