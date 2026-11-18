import unittest
from pathlib import Path
import torch
from exposure_market import Market,evaluate
from numeric_policy import NumericPolicy
from train_direct import collect
from direct_ledger import negative_sharpe

class GlobalCollectionTests(unittest.TestCase):
    def test_full_path_matches_evaluation_and_has_finite_gradients(self):
        data=Path(__file__).resolve().parents[1]/'sector_data_20261107/sector9_development_through2024.csv'
        market=Market(data,'2001-01-01','2001-12-31');policy=NumericPolicy(market.env.tickers)
        segment=collect(policy,market);expected=evaluate(policy,market,'cpu')
        self.assertEqual(len(segment[3]),expected['metrics']['steps'])
        self.assertEqual(len(segment[0]),len(expected['probabilities']))
        objective=negative_sharpe(segment[3]);self.assertAlmostEqual(-objective.item(),expected['metrics']['sharpe_proxy'],places=11)
        gradient=torch.autograd.grad(objective,segment[2])[0]
        self.assertTrue(torch.isfinite(gradient).all());self.assertGreater(gradient.abs().sum().item(),0.)

if __name__=='__main__':unittest.main()
