import unittest
import numpy as np
from strategy_ppo import Market

class MacroTests(unittest.TestCase):
    def test_drift_above_target_cap_is_not_an_unrequested_trade(self):
        m=Market('../sector_data_20261107/sector9_development_through2024.csv','2010-01-01','2010-12-31')
        m.reset();i=m.env.index;m.env.prices=m.env.prices.copy()
        m.env.prices[i+1:i+6]=m.env.prices[i]
        m.env.prices[i+1:i+6,0]*=2
        target=np.full(9,.7/8);target[0]=.3
        m.step(target)
        self.assertAlmostEqual(m.env.equity,100000*.999*1.3,places=4)
        self.assertGreater(m.env.weights[0],.3)
        self.assertAlmostEqual(sum(r['turnover'] for r in m.infos[1:]),0.,places=7)
    def test_lag_does_not_use_execution_close(self):
        m=Market('../sector_data_20261107/sector9_development_through2024.csv','2010-01-01','2010-12-31')
        old,targets=m.reset();i=m.env.index;m.env.prices=m.env.prices.copy();m.env.prices[i:]*=3;m.cache={}
        obs,new=m.observe();np.testing.assert_array_equal(old['market'],obs['market']);np.testing.assert_array_equal(targets,new)

if __name__=='__main__':unittest.main()
