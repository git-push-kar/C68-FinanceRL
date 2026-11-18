from typing import Literal
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from finance import paper_engine as engine
from finance.policy import predict
from pathlib import Path

router = APIRouter(prefix='/finance/paper', tags=['finance-paper'])
ASSESSMENT_MARKER = Path(__file__).resolve().parents[1]/'finance/assessment_active.json'


class Start(BaseModel):
    capital: float = Field(default=100000., ge=100, le=10000000, allow_inf_nan=False)
    mode: Literal['benchmark', 'ytd', 'live', 'historical'] = 'benchmark'


class Advance(BaseModel):
    mode: Literal['day', 'rebalance'] = 'day'
    expected_sessions: int = Field(default=0, ge=0)


class SettleRequest(BaseModel):
    target_date: str = Field(default='', description='Optional forced ISO date for settling')


@router.get('/config')
def config():
    m = engine.panel().manifest
    return {
        **{k: m[k] for k in ['title', 'currency', 'first_return', 'last_return', 'adapter',
                             'rebalance_sessions', 'fee_bps', 'research_note', 'adapter_sha256']},
        'conversation_enabled': False,
        'supported_modes': ['benchmark', 'ytd', 'live']
    }


@router.post('/sessions')
def start(request: Start):
    if request.mode == 'live':
        from finance.live_manager import get_live_realtime_session
        mgr = get_live_realtime_session(request.capital)
        mgr.refresh_quotes()
        return {'session_id': 'live_market_session', **mgr.snapshot()}

    if request.mode == 'ytd':
        from finance.live_manager import get_ytd_session
        mgr = get_ytd_session(request.capital)
        mgr.reset()
        return {'session_id': 'ytd_history_session', **mgr.snapshot()}

    try:
        key, session = engine.create(request.capital)
        return {'session_id': key, **session.snapshot()}
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc


@router.get('/sessions/{key}')
def snapshot(key: str):
    if key == 'live_market_session':
        from finance.live_manager import get_live_realtime_session
        mgr = get_live_realtime_session()
        return {'session_id': key, **mgr.snapshot()}

    if key == 'ytd_history_session':
        from finance.live_manager import get_ytd_session
        mgr = get_ytd_session()
        return {'session_id': key, **mgr.snapshot()}

    try:
        session = engine.account(key)
        with session.lock:
            return {'session_id': key, **session.snapshot()}
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc


@router.post('/sessions/{key}/advance')
def advance(key: str, request: Advance):
    if key == 'live_market_session':
        from finance.live_manager import get_live_realtime_session
        mgr = get_live_realtime_session()
        return {'session_id': key, **mgr.execute_live_decision()}

    if key == 'ytd_history_session':
        from finance.live_manager import get_ytd_session
        mgr = get_ytd_session()
        mgr.step(mode=request.mode)
        return {'session_id': key, **mgr.snapshot()}

    from api.server import get_service
    if ASSESSMENT_MARKER.exists():
        raise HTTPException(503, 'The portfolio model is being evaluated. Please retry shortly.')
    try:
        session = engine.account(key)
        with session.lock:
            if request.expected_sessions != session.n:
                raise HTTPException(409, 'The portfolio has already advanced. Reload its current state.')
            before = len(session.activities)
            def _call_predict():
                try:
                    return predict(get_service(), account=session)
                except TypeError:
                    return predict(get_service())

            for _ in range(5 if request.mode == 'rebalance' else 1):
                if not session.step(_call_predict):
                    break
                if request.mode == 'rebalance' and len(session.activities) > before:
                    break
            return {'session_id': key, **session.snapshot()}
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(503, f'Paper trading stopped: {exc}. No substitute model decision was used.') from exc


@router.post('/live/refresh')
def live_refresh():
    from finance.live_manager import get_live_realtime_session
    mgr = get_live_realtime_session()
    return {'session_id': 'live_market_session', **mgr.refresh_quotes()}


@router.post('/live/decide')
def live_decide():
    from finance.live_manager import get_live_realtime_session
    mgr = get_live_realtime_session()
    return {'session_id': 'live_market_session', **mgr.execute_live_decision()}


@router.post('/live/settle')
def live_settle(req: SettleRequest = SettleRequest()):
    from finance.live_manager import get_live_realtime_session
    mgr = get_live_realtime_session()
    target_d = req.target_date.strip() if req.target_date else None
    return {'session_id': 'live_market_session', **mgr.settle_today(target_date=target_d)}


@router.delete('/sessions/{key}')
def close(key: str):
    if key in ('live_market_session', 'ytd_history_session'):
        return {'closed': True}
    engine.close(key)
    return {'closed': True}
