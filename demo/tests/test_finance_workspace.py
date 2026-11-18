import unittest
from unittest.mock import patch
from contextlib import contextmanager
from fastapi.testclient import TestClient
from api.server import app
from finance.service import evidence
from finance.grounding import context,messages
from finance.generation import generate

class FinanceWorkspaceTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):cls.client=TestClient(app)
 def test_all_daily_accounts_reconcile(self):
  for s in evidence()['sessions'].values():
   self.assertLess(s['parity_error'],1e-11)
   for d in s['daily']:
    self.assertAlmostEqual(sum(d['asset_values'])+d['equity']*d['cash_weight'],d['equity'],places=6)
    self.assertAlmostEqual(100000+sum(d['cumulative_asset_market_pnl'])-d['cumulative_fees'],d['equity'],places=6)
 def test_all_trade_funding_and_contributions(self):
  for s in evidence()['sessions'].values():
   for r in s['decisions']:
    self.assertLess(r['signal_date'],r['execution_date'])
    self.assertAlmostEqual(sum(r['notional_changes'])+r['fee'],r['equity_before']*r['cash_before'],places=6)
    for i in range(5):self.assertAlmostEqual(sum(x[i] for x in r['strategy_contributions']),r['target_weights'][i],places=10)
 def test_context_actual_trade_and_scaled_capital(self):
  f,idx,sources=context(42,5,50000,25,'Why bonds?');t=f['selected_activity']['trades'][2]
  self.assertEqual(t['action'],'buy');self.assertEqual(t['amount_usd'],280.37)
  self.assertEqual(idx,5);self.assertTrue(sources)
 def test_risk_context_is_causal_and_contributions_reconcile(self):
  for s in evidence()['sessions'].values():
   for row in s['decisions']:
    risk=row['risk_evidence']
    self.assertEqual(risk['signal_date'],row['signal_date'])
    self.assertLess(risk['signal_date'],row['execution_date'])
    self.assertAlmostEqual(sum(risk['target_variance_contribution_fraction']),1,places=9)
    self.assertGreater(risk['target_annualized_volatility'],0)
 def test_context_no_future_outcomes(self):
  f,_,_=context(42,0,100000,0);self.assertEqual(f['selected_activity']['retrospective_return_end'],'2026-02-02')
  self.assertNotIn('final_research_comparison',f)
  with self.assertRaises(ValueError):context(42,1,100000,4)
 def test_history_is_passed_to_model_not_intent_templates(self):
  f,_,_=context(42,5,100000,25)
  history=[{'role':'user','content':'What about TLT?'},{'role':'assistant','content':'We are discussing Treasuries.'}]
  m=messages('What does that mean for me?',history,f)
  self.assertEqual(m[1:3],history);self.assertIn('What does that mean for me?',m[-1]['content'])
  self.assertIn('TLT CURRENT HOLDING',m[-1]['content'])
 def test_stub_backend_cannot_answer_finance(self):
  class FakeService:
   @contextmanager
   def lease(self,task):yield {'mode':'stub'}
  with self.assertRaisesRegex(RuntimeError,'actual model generation'):
   list(generate(FakeService(),'Why?',42,0,100000,0))
 def test_removed_explanations_never_invoke_generator(self):
  with patch('api.finance.generate') as mocked:
   for endpoint in ['/finance/explain','/finance/explain/stream']:
    r=self.client.post(endpoint,json={'question':'Why?','decision_id':0,'capital':100000})
    self.assertEqual(r.status_code,410)
    self.assertIn('removed after factual-accuracy testing',r.text)
   mocked.assert_not_called()
 def test_invalid_requests(self):
  r=self.client.post('/finance/explain/stream',json={'question':'Why?','decision_id':0,'capital':-1})
  self.assertEqual(r.status_code,422)
 def test_original_tabs_and_finance_reachable(self):
  r=self.client.get('/');self.assertEqual(r.status_code,200)
  for name in ['tab-chat','tab-anomaly','tab-finance']:self.assertIn(name,r.text)
  self.assertEqual(self.client.get('/ui/finance.html').status_code,200)
  self.assertEqual(self.client.get('/finance/replay').status_code,200)

if __name__=='__main__':unittest.main()
