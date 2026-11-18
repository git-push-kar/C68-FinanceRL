import unittest
import numpy as np
import torch
from cash_execution import execute
from direct_ledger import step,path_returns,negative_sharpe

class DirectLedgerTests(unittest.TestCase):
    def test_random_ledger_parity(self):
        rng=np.random.default_rng(71)
        for _ in range(100):
            w=rng.dirichlet(np.ones(4));target=rng.dirichlet(np.ones(4))[:3];rel=np.exp(rng.normal(0,.02,3))
            reference=execute(w[:3],w[3],target,rel)
            actual=step(*[torch.tensor(x,dtype=torch.float64) for x in (w[:3],w[3],target,rel)])
            np.testing.assert_allclose(actual[0].numpy(),reference.stock_weights,atol=1e-13,rtol=0)
            np.testing.assert_allclose([x.item() for x in actual[1:]],[reference.cash_weight,reference.net_return,reference.trading_cost],atol=1e-13,rtol=0)
    def test_fee_gradient(self):
        exposure=torch.tensor(.71,dtype=torch.float64,requires_grad=True)
        def f(x):return step(torch.tensor([.1,.3],dtype=torch.float64),torch.tensor(.6,dtype=torch.float64),x*torch.ones(2,dtype=torch.float64)/2,torch.tensor([1.02,.97],dtype=torch.float64))[2]
        torch.autograd.gradcheck(f,(exposure,),eps=1e-6,atol=1e-7,rtol=1e-5)
    def test_whole_path_gradient_and_parity(self):
        rng=np.random.default_rng(45);rel=np.exp(rng.normal(.0002,.012,(26,3)))
        exposures=torch.tensor([.42,.77,.51,.83,.36,.61],dtype=torch.float64,requires_grad=True)
        def f(x):return negative_sharpe(path_returns(x,np.array([.13,.18,.19]),.5,rel))
        torch.autograd.gradcheck(f,(exposures,),eps=1e-6,atol=1e-6,rtol=1e-4)
        expected=[];w=np.array([.13,.18,.19]);cash=.5
        for d,r in enumerate(rel):
            target=exposures[d//5].item()*np.ones(3)/3 if d%5==0 else w.copy()
            actual=execute(w,cash,target,r);w=actual.stock_weights;cash=actual.cash_weight;expected.append(actual.net_return)
        np.testing.assert_allclose(path_returns(exposures,[.13,.18,.19],.5,rel).detach().numpy(),expected,atol=1e-13,rtol=0)
    def test_two_pass_parameter_gradient(self):
        torch.manual_seed(42);model=torch.nn.Linear(2,1).double();obs=torch.randn(6,2,dtype=torch.float64)
        rng=np.random.default_rng(3);rel=np.exp(rng.normal(.0004,.01,(30,3)))
        exposures=.25+.75*model(obs).sigmoid().flatten()
        negative_sharpe(path_returns(exposures,[0.,0.,0.],1.,rel)).backward()
        expected=[p.grad.clone() for p in model.parameters()];model.zero_grad()
        leaves=exposures.detach().clone().requires_grad_()
        coefficients=torch.autograd.grad(negative_sharpe(path_returns(leaves,[0.,0.,0.],1.,rel)),leaves)[0]
        for x,c in zip(obs,coefficients):((.25+.75*model(x).sigmoid())*c).backward()
        for p,g in zip(model.parameters(),expected):torch.testing.assert_close(p.grad,g,rtol=1e-12,atol=1e-12)

if __name__=='__main__':unittest.main()
