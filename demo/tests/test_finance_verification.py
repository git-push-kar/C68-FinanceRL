import unittest
from finance.verification import issues
from finance.grounding import context
class FinanceDraftChecks(unittest.TestCase):
 def setUp(self):self.facts=context(42,37,100000,188)[0]
 def test_negative_comparison_is_flagged(self):self.assertTrue(issues('GLD had the higher contribution.',self.facts,'','Compare GLD and TLT profit'))
 def test_correct_negative_comparison_is_not_flagged(self):self.assertFalse(issues('TLT had the higher contribution.',self.facts,'','Compare GLD and TLT profit'))
 def test_unsupported_number_is_flagged(self):self.assertTrue(issues('92.55% of allocations',self.facts,'Sharpe 2.170','Compare Sharpe'))
 def test_rounding_is_supported(self):self.assertFalse(issues('Sharpe 2.170',self.facts,'Sharpe 2.169516','Compare Sharpe'))
 def test_benchmark_omission_is_flagged(self):self.assertTrue(issues('Trained Sharpe 1.951',self.facts,'1.951','Why use AI instead of minimum variance?'))
