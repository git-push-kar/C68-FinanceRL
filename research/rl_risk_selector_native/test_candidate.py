import unittest
import numpy as np
import torch
from candidate_market import Market
from candidate_path import path_returns
from numeric_policy import NumericPolicy
from train_direct import collect
from direct_ledger import negative_sharpe

class CandidateTests(unittest.TestCase):
    def test_full_market_reconstruction(self):
        torch.set_num_threads(2)
        m=Market('../multiasset_data_20261115/five_asset_development_through2024.csv','2019-01-01','2021-12-31')
        policy=NumericPolicy(m.env.tickers)
        obs,old,leaves,returns,error=collect(policy,m)
        self.assertLess(error,1e-11)
        self.assertEqual(len(returns),len(m.infos))
        self.assertTrue(torch.isfinite(torch.autograd.grad(negative_sharpe(returns),leaves)[0]).all())

    def test_probability_gradient(self):
        rng=np.random.default_rng(18);targets=rng.dirichlet(np.ones(7),size=(3,5))
        rel=1+rng.normal(.001,.02,size=(13,7));p=rng.dirichlet(np.ones(5),size=3)
        leaves=torch.tensor(p,requires_grad=True)
        def objective(x):return negative_sharpe(path_returns(x,targets,np.zeros(7),1.,rel))
        g=torch.autograd.grad(objective(leaves),leaves)[0].numpy()
        for i in range(3):
            for j in range(5):
                a=p.copy();b=p.copy();a[i,j]+=1e-6;b[i,j]-=1e-6
                fd=float((objective(torch.tensor(a))-objective(torch.tensor(b)))/2e-6)
                self.assertAlmostEqual(g[i,j],fd,places=5)

    def test_future_prices_do_not_change_observation(self):
        m=Market('../multiasset_data_20261115/five_asset_development_through2024.csv','2019-01-01','2021-12-31');before=m.observe()
        m.env.prices[m.env.index:]*=1.7;m.cache.clear();after=m.observe()
        np.testing.assert_array_equal(before[0]['summary'],after[0]['summary'])
        np.testing.assert_array_equal(before[1],after[1])

if __name__=='__main__':unittest.main()
