import unittest
import numpy as np
from cash_execution import execute

class CashAccountingTests(unittest.TestCase):
    def test_cash_stays_cash(self):
        r=execute([0.,0.],1.,[0.,0.],[.8,1.2])
        self.assertEqual(r.cash_weight,1.);self.assertEqual(r.net_return,0.);self.assertEqual(r.trading_cost,0.)
    def test_buy_uses_exact_fee_adjusted_notional(self):
        r=execute([0.],1.,[1.],[1.]);k=.001
        self.assertAlmostEqual(r.trading_cost,k/(1+k),places=14)
        self.assertAlmostEqual(1+r.net_return,1/(1+k),places=14)
        self.assertAlmostEqual(r.stock_weights[0],1.)
    def test_sold_stock_no_longer_receives_market_return(self):
        r=execute([1.],0.,[0.],[2.])
        self.assertAlmostEqual(r.net_return,-.001,places=14);self.assertEqual(r.cash_weight,1.)
    def test_holding_drifts_without_trades(self):
        r=execute([.4,.4],.2,[.4,.4],[1.1,.9])
        self.assertEqual(r.trading_cost,0.);self.assertEqual(r.turnover,0.)
        np.testing.assert_allclose(r.stock_weights,[.44,.36]);self.assertAlmostEqual(r.cash_weight,.2)
    def test_partial_exposure_and_cash_interest(self):
        r=execute([0.],1.,[.5],[1.1],cash_relative=1.01)
        cost=.0005/1.0005
        self.assertAlmostEqual(1+r.net_return,(1-cost)*1.055,places=14)
    def test_round_trip(self):
        buy=execute([0.],1.,[1.],[1.]);sell=execute(buy.stock_weights,buy.cash_weight,[0.],[1.])
        self.assertAlmostEqual((1+buy.net_return)*(1+sell.net_return),.999/1.001,places=14)
    def test_random_self_financing_invariants(self):
        rng=np.random.default_rng(42)
        for _ in range(200):
            old=rng.dirichlet(np.ones(10));target=rng.dirichlet(np.ones(10));rel=rng.uniform(.8,1.2,9)
            r=execute(old[:-1],old[-1],target[:-1],rel)
            self.assertAlmostEqual(r.trading_cost,.001*r.turnover,places=14)
            cash_after_trade=old[-1]-(r.post_fee_stock_notional-old[:-1]).sum()-r.trading_cost
            self.assertAlmostEqual(cash_after_trade,(1-r.trading_cost)*target[-1],places=14)
            self.assertAlmostEqual(r.stock_weights.sum()+r.cash_weight,1.,places=14)
            self.assertAlmostEqual(1+r.net_return,(1-r.trading_cost)*(1+r.gross_return),places=14)
    def test_invalid_portfolios_rejected(self):
        for args in [([1.],1.,[1.],[1.]),([0.],1.,[1.1],[1.]),([0.],1.,[-.1],[1.]),([0.],1.,[.5],[0.])]:
            with self.assertRaises(ValueError):execute(*args)

if __name__=='__main__':unittest.main()
