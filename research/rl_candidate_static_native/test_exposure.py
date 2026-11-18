import copy,unittest
from pathlib import Path
import numpy as np
import torch
from exposure_market import Market,evaluate,tensor
from counterfactual import action_values
from numeric_policy import NumericPolicy

DATA=Path(__file__).resolve().parents[1]/'sector_data_20261107/sector9_development_through2024.csv'

class ExposureTests(unittest.TestCase):
    def setUp(self):self.m=Market(DATA,'2001-01-01','2001-03-01')
    def test_signal_causality(self):
        obs,_=self.m.reset();other=copy.deepcopy(self.m);i=other.env.index
        other.env.prices[i:]*=1.7;other.env.basket[i:]*=1.4;other.cache={}
        np.testing.assert_array_equal(obs['summary'],other.observe()[0]['summary'])
    def test_no_off_day_trading_and_conservation(self):
        _,targets=self.m.reset();self.m.step(targets[1])
        self.assertTrue(all(x['turnover']==0 for x in self.m.infos[1:]))
        self.assertAlmostEqual(self.m.env.weights.sum()+self.m.env.cash_weight,1.,places=13)
        for row in self.m.infos:
            self.assertLess(row['signal_close_date'],row['decision_date'])
            self.assertAlmostEqual((1-row['trading_cost'])*(1+row['gross_return'])-1,row['net_return'],places=13)
    def test_counterfactual_isolation_and_reward(self):
        _,targets=self.m.reset();before=copy.deepcopy(self.m.env)
        values=action_values(self.m,targets,lambda obs:np.ones(4)/4,[])
        for a in range(4):
            other=copy.deepcopy(self.m);_,reward,_=other.step(targets[a])
            self.assertAlmostEqual(float(values[a]),reward,places=5)
        np.testing.assert_array_equal(before.weights,self.m.env.weights)
        self.assertEqual(before.index,self.m.env.index);self.assertEqual(self.m.infos,[])
    def test_uniform_policy_matches_fixed_exposure(self):
        torch.manual_seed(42);p=NumericPolicy(self.m.env.tickers)
        x=evaluate(p,self.m,'cpu');y=evaluate(None,self.m,'cpu',fixed=.625)
        np.testing.assert_allclose([d['net_return'] for d in x['daily']],[d['net_return'] for d in y['daily']],atol=1e-14)
    def test_real_update_and_reload(self):
        import tempfile
        p=NumericPolicy(self.m.env.tickers);obs=tensor(self.m.reset()[0],'cpu')
        optimizer=torch.optim.Adam(p.parameters(),lr=.001)
        loss=-p(obs)[0].probs[0,0];loss.backward();optimizer.step()
        self.assertGreater(p(obs)[0].probs[0,0].item(),.25)
        with tempfile.TemporaryDirectory() as d:
            p.save(d,{});loaded=NumericPolicy(self.m.env.tickers,d)
            torch.testing.assert_close(p(obs)[0].probs,loaded(obs)[0].probs,rtol=0,atol=0)

if __name__=='__main__':unittest.main()
