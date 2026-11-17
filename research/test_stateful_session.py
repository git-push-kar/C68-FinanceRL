"""Accounting parity tests with explicit synthetic policy; not real-model proof."""
import sys,copy
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).parent/'stateful_rl_20261117_run4'))
sys.path.insert(0,str(Path(__file__).parent))
from market import Market,evaluate
from stateful_session import StatefulSession

def example_policy(observation):
 # A deterministic fixture that depends on changing observations. Production
 # must inject native policy.choose instead; this is not a demo fallback.
 actions=np.array([5 if s['return60']>0 else 3 for s in observation['stocks']])
 return actions,{'mode':'explicit_test_fixture'}

@pytest.mark.parametrize('capital',[500.,5000.,100000.])
def test_daily_session_exactly_reproduces_research_accounting(capital):
 market=Market();expected=evaluate(market,example_policy,'2024-01-01','2024-12-31',capital)
 session=StatefulSession(market,example_policy,'2024-01-01','2024-12-31',capital)
 while session.advance():pass
 assert np.allclose(session.values,expected['daily_equity'],rtol=0,atol=1e-8)
 assert abs(session.fees-expected['metrics']['fees'])<1e-8
 assert abs(session.slippage-expected['metrics']['slippage'])<1e-8
 assert session.model_calls==len(expected['decisions'])
 for actual,reference in zip(session.decisions,expected['decisions']):
  assert actual['observation']==reference['observation']
  assert actual['actions']==reference['actions'] and actual['trade_shares']==reference['trade_shares']

def test_inference_failure_is_atomic_and_snapshot_cannot_mutate_account():
 def broken(obs):raise RuntimeError('Explicit inference failure')
 session=StatefulSession(Market(),broken,'2024-01-01','2024-12-31',100000.)
 before=copy.deepcopy(session.snapshot())
 with pytest.raises(RuntimeError):session.advance()
 assert session.snapshot()==before
 snapshot=session.snapshot();snapshot['holdings'][0]['price']=0;snapshot['daily_equity'].append(0)
 assert session.snapshot()==before

def test_no_extra_call_when_account_is_complete():
 session=StatefulSession(Market(),example_policy,'2024-01-01','2024-01-10',100000.)
 while session.advance():pass
 before=copy.deepcopy(session.snapshot());assert not session.advance();assert session.snapshot()==before
