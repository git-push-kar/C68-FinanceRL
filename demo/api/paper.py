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


class Advance(BaseModel):
    mode: Literal['day', 'rebalance'] = 'day'
    expected_sessions: int = Field(ge=0)


@router.get('/config')
def config():
    m = engine.panel().manifest
    return {**{k: m[k] for k in ['title', 'currency', 'first_return', 'last_return', 'adapter',
                                 'rebalance_sessions', 'fee_bps', 'research_note', 'adapter_sha256']},
            'conversation_enabled': False}


@router.post('/sessions')
def start(request: Start):
    try:
        key, session = engine.create(request.capital)
        return {'session_id': key, **session.snapshot()}
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc


@router.get('/sessions/{key}')
def snapshot(key: str):
    try:
        session = engine.account(key)
        with session.lock:
            return {'session_id': key, **session.snapshot()}
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc


@router.post('/sessions/{key}/advance')
def advance(key: str, request: Advance):
    from api.server import get_service
    if ASSESSMENT_MARKER.exists():
        raise HTTPException(503, 'The portfolio model is being evaluated. Please retry shortly.')
    try:
        session = engine.account(key)
        with session.lock:
            if request.expected_sessions != session.n:
                raise HTTPException(409, 'The portfolio has already advanced. Reload its current state.')
            before = len(session.activities)
            for _ in range(5 if request.mode == 'rebalance' else 1):
                if not session.step(lambda: predict(get_service())):
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


@router.delete('/sessions/{key}')
def close(key: str):
    engine.close(key)
    return {'closed': True}
