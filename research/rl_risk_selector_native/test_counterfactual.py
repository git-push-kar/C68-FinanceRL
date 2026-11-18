import copy,unittest
import numpy as np
import torch
from counterfactual import action_values
from strategy_ppo import Market

class CounterfactualTests(unittest.TestCase):
    def market(self):
        return Market('../sector_data_20261107/sector9_development_through2024.csv','2010-01-01','2010-12-31')
    def test_rollouts_do_not_mutate_actual_portfolio(self):
        m=self.market();obs,targets=m.reset();m.step(targets[2]);obs,targets=m.observe()
        before=copy.deepcopy(m.env);known=obs['market'].copy();cache_keys=set(m.cache)
        q=action_values(m,targets,lambda obs:np.ones(4)/4,[.2,.8])
        self.assertTrue(np.isfinite(q).all());self.assertEqual(m.env.index,before.index)
        self.assertEqual(m.env.equity,before.equity);self.assertEqual(m.env.returns,before.returns)
        np.testing.assert_array_equal(m.env.weights,before.weights)
        np.testing.assert_array_equal(m.env.benchmark_weights,before.benchmark_weights)
        np.testing.assert_array_equal(m.observe()[0]['market'],known)
        self.assertEqual(set(m.cache),cache_keys)
    def test_one_decision_matches_actual_environment_reward(self):
        m=self.market();obs,targets=m.reset()
        q=action_values(m,targets,None,[])
        for a in range(4):
            m.reset();_,reward,_=m.step(targets[a]);self.assertAlmostEqual(float(q[a]),reward,places=6)
    def test_future_changes_reward_but_not_current_actor_input(self):
        m=self.market();obs,targets=m.reset();q=action_values(m,targets,None,[])
        m.env.prices=m.env.prices.copy();m.env.prices[m.env.index+1:,0]*=1.1;m.cache={}
        after,new_targets=m.observe();new_q=action_values(m,new_targets,None,[])
        np.testing.assert_array_equal(obs['market'],after['market'])
        np.testing.assert_array_equal(targets,new_targets)
        self.assertGreater(float(np.max(abs(q-new_q))),.01)
    def test_expected_reward_gradient_has_correct_sign_and_baseline_invariance(self):
        logits=torch.tensor([.2,-.1,.3,.0],dtype=torch.float64,requires_grad=True)
        q=torch.tensor([.1,-.2,.4,.0],dtype=torch.float64);p=logits.softmax(0)
        loss=-(p*(q-q.mean())).sum();loss.backward()
        expected=-p.detach()*(q-(p.detach()*q).sum())
        torch.testing.assert_close(logits.grad,expected,atol=1e-12,rtol=1e-12)

if __name__=='__main__':unittest.main()
