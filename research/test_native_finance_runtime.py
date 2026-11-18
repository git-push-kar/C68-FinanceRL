"""CPU execution-contract tests; not a substitute for a real native reload gate."""
from copy import deepcopy
from types import SimpleNamespace
import unittest
import numpy as np
import torch
from native_finance_runtime import NativeFinanceSpecialist


class PolicyDouble:
    def __init__(self, actions=4, features=27):
        self.tickers = ['GLD', 'SPY', 'TLT', 'UUP', 'XLE']
        self.shared = SimpleNamespace(model=object(), compute_device='cpu')
        self.calls = 0
        self.actions, self.features = actions, features

    def __call__(self, obs):
        self.calls += 1
        assert obs['summary'].shape == (1, self.features)
        probs = [.1, .2, .3, .4] if self.actions == 4 else [.05, .1, .15, .2, .5]
        return torch.distributions.Categorical(probs=torch.tensor([probs])), None


class ExecutionContract(unittest.TestCase):
    def setUp(self):
        self.specialist = NativeFinanceSpecialist.__new__(NativeFinanceSpecialist)
        self.specialist.policy = PolicyDouble()
        self.specialist.cap = .6
        self.specialist.action_count, self.specialist.feature_count = 4, 27
        self.specialist.schema = 'multiasset_summary_text_v1'
        targets = .1 + .5*np.eye(5)[:4]
        self.payload = {'tickers': self.specialist.policy.tickers,
                        'trading_step': 0, 'execution_portfolio': [0, 0, 0, 0, 0, 1],
                        'signal_lag_closes': 1, 'signal_close_date': '2024-01-05',
                        'execution_close_date': '2024-01-08',
                        'observation': {'summary': [0]*27, 'candidate_targets': targets.tolist()}}

    def predict(self, payload=None):
        return self.specialist.predict(self.specialist.policy.shared.model, payload or self.payload)

    def test_rebalance_uses_expected_mixture(self):
        result = self.predict()
        np.testing.assert_allclose(result['weights'], [.15, .2, .25, .3, .1], atol=1e-8, rtol=0)
        self.assertTrue(result['rebalance_due'])
        self.assertEqual(result['cash_weight'], 0)

    def test_fifth_risk_strategy_is_included_in_execution(self):
        self.specialist.policy = PolicyDouble(5, 33)
        self.specialist.action_count, self.specialist.feature_count = 5, 33
        self.specialist.schema = 'risk_selector_static_text_v1'
        self.payload['observation']['summary'] = [0]*33
        self.payload['observation']['candidate_targets'].append([.105, .105, .105, .105, .58])
        result = self.predict()
        np.testing.assert_allclose(result['weights'], [.1275, .1525, .1775, .2025, .34], atol=2e-8, rtol=0)
        self.assertEqual(len(result['probabilities']), 5)

    def test_off_schedule_preserves_drift_and_cash_without_model_call(self):
        payload = {'tickers': self.specialist.policy.tickers, 'trading_step': 1,
                   'execution_portfolio': [.7, .05, .05, .05, .05, .1]}
        result = self.predict(payload)
        self.assertEqual(result['weights'], payload['execution_portfolio'][:-1])
        self.assertEqual(result['cash_weight'], .1)
        self.assertFalse(result['rebalance_due'])
        self.assertEqual(self.specialist.policy.calls, 0)

    def test_bad_execution_inputs_rejected_before_policy_call(self):
        cases = []
        for key, value in [('signal_close_date', '2024-01-08'), ('signal_lag_closes', True),
                           ('trading_step', True), ('execution_portfolio', [.2]*6),
                           ('tickers', list(reversed(self.payload['tickers'])))]:
            p = deepcopy(self.payload);p[key] = value;cases.append(p)
        p = deepcopy(self.payload);p['observation']['summary'][2] = float('nan');cases.append(p)
        p = deepcopy(self.payload);p['observation']['summary'][2] = 1.234;cases.append(p)
        p = deepcopy(self.payload);p['observation']['candidate_targets'][0] = [1, 0, 0, 0, 0];cases.append(p)
        for payload in cases:
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                self.predict(payload)
        self.assertEqual(self.specialist.policy.calls, 0)


if __name__ == '__main__':
    unittest.main()
