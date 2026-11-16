import unittest
import numpy as np
from strategy_ppo import Market

class AccountingTests(unittest.TestCase):
    def setUp(self):self.m=Market('../data/nifty50_prices.csv','2022-01-01','2022-12-31')
    def test_delayed_observation_cannot_see_execution_close(self):
        m=self.m;obs,target=m.reset();i=m.env.index
        m.env.prices=m.env.prices.copy();m.env.prices[i:]*=1.7;m.cache={}
        changed,new_target=m.observe()
        np.testing.assert_array_equal(obs['market'],changed['market'])
        np.testing.assert_array_equal(target,new_target)
    def test_targets_valid_and_first_week_accounting(self):
        m=self.m;obs,targets=m.reset();i=m.env.index;w=targets[2]
        self.assertTrue(np.all(targets>=0));self.assertLessEqual(targets.max(),.30)
        np.testing.assert_allclose(targets.sum(1),1)
        _,reward,done=m.step(w)
        expected=100000*(1-.001)*float(w@(m.env.prices[i+5]/m.env.prices[i]))
        self.assertAlmostEqual(m.env.equity,expected,places=4)
        self.assertEqual(len(m.infos),5)
        self.assertAlmostEqual(sum(x['turnover'] for x in m.infos),1.,places=6)
        self.assertTrue(all(x['signal_close_date']<x['decision_date']<x['return_date'] for x in m.infos))
    def test_reset_restores_actual_cash(self):
        m=self.m;_,targets=m.reset();m.step(targets[0]);obs,_=m.reset()
        self.assertEqual(obs['portfolio'][-1],1.)
        self.assertEqual(obs['portfolio'][:-1].sum(),0.)

if __name__=='__main__':unittest.main()
