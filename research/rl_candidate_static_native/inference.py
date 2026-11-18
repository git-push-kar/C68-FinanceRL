"""Load finance beside other specialists on ONE resident InternVL3 model."""
import argparse
import json
from datetime import date
from pathlib import Path

import numpy as np
import torch

from allocation import allocation_weights
from config import BACKBONE_ID, BACKBONE_REVISION, ModelConfig
from model import InternVLFinancePolicy, validate_package, write_json
from shared_backbone import DomainRouter, SharedBackbone
from execution_policy import configure_finance_precision


class FinanceSpecialist:
    def __init__(self, package_dir, shared):
        self.policy = InternVLFinancePolicy.load_artifacts(package_dir, shared=shared)
        self.environment = json.loads((Path(package_dir) / "environment_config.json").read_text())

    def register(self, router):
        if router.shared is not self.policy.shared:
            raise ValueError("Finance and router must reference the same resident backbone")
        router.register("finance", "finance", self.predict)

    def predict(self, model, payload):
        if self.policy.cfg.dtype=='fp32':configure_finance_precision()
        if model is not self.policy.shared.model:
            raise ValueError("Finance handler received a different model instance")
        if payload.get("tickers") != self.policy.tickers:
            raise ValueError("Supply the exact trained ticker order with every finance request")
        obs = payload["observation"]
        assets, window = len(self.policy.tickers), self.policy.cfg.window
        shapes = {"market": (window, assets, 5), "portfolio": (assets + 1,), "risk": (3,)}
        if self.policy.cfg.covariance_window:
            shapes['covariance']=(assets,assets)
        device = self.policy.shared.compute_device
        tensors = {}
        for key, shape in shapes.items():
            array = np.asarray(obs[key], dtype=np.float32)
            if array.shape != shape or not np.isfinite(array).all():
                raise ValueError(f"Observation {key} must have finite values and shape {shape}")
            if key=='covariance':
                if not np.allclose(array,array.T,rtol=1e-4,atol=1e-8) or (np.diag(array)<=0).any():
                    raise ValueError('Covariance must be symmetric with positive diagonal')
                if np.linalg.eigvalsh(array).min() < -1e-5*np.diag(array).max():
                    raise ValueError('Covariance must be positive semidefinite')
            tensors[key] = torch.as_tensor(array, device=device).unsqueeze(0)
        holdings = tensors["portfolio"][0]
        if (holdings < 0).any() or not torch.isclose(holdings.sum(), torch.tensor(1., device=device), atol=1e-5):
            raise ValueError("Portfolio weights, including cash, must be nonnegative and sum to one")
        lag=payload.get('signal_lag_closes',0)
        if not isinstance(lag,int) or isinstance(lag,bool) or lag not in (0,1):raise ValueError('signal_lag_closes must be zero or one')
        if lag:
            if 'execution_portfolio' not in payload:
                raise ValueError('Delayed signals require actual current execution_portfolio, separate from policy state')
            try:
                signal_date=date.fromisoformat(payload['signal_close_date'])
                execution_date=date.fromisoformat(payload['execution_close_date'])
            except (KeyError,TypeError,ValueError) as error:
                raise ValueError('Delayed signals require ISO signal_close_date and execution_close_date') from error
            if signal_date>=execution_date:raise ValueError('Signal close must precede execution close')
        if 'execution_portfolio' in payload:
            current=np.asarray(payload['execution_portfolio'],dtype=np.float32)
            if current.shape!=(assets+1,) or not np.isfinite(current).all() or (current<0).any() or not np.isclose(current.sum(),1.,atol=1e-5):
                raise ValueError('execution_portfolio must be current finite nonnegative asset/cash weights summing to one')
            holdings=torch.as_tensor(current,device=device)
        with torch.inference_mode():
            scores, value = self.policy(tensors)
        interval=self.environment['reward'].get('rebalance_interval',1)
        if interval>1:
            step=payload.get('trading_step')
            if not isinstance(step,int) or isinstance(step,bool) or step<0:
                raise ValueError('Scheduled policy requires trading_step: zero at the first decision, increment once per trading day')
            rebalance_due=step%interval==0
        else:rebalance_due=True
        weights = (allocation_weights(scores[0].cpu().numpy(), self.environment["reward"]["max_weight"],
                                     holdings[:-1].cpu().numpy(), self.environment["reward"].get("rebalance_rate", 1.0))
                   if rebalance_due else holdings[:-1].cpu().numpy())
        return {"domain": "finance", "base_model_id": BACKBONE_ID, "base_revision": BACKBONE_REVISION,
                "tickers": self.policy.tickers, "weights": weights.tolist(),
                "cash_weight": 0.0 if rebalance_due else float(holdings[-1]),'rebalance_due':rebalance_due,
                'signal_lag_closes':lag,'signal_close_date':payload.get('signal_close_date'),
                'execution_close_date':payload.get('execution_close_date'),
                "state_value": None if self.policy.cfg.covariance_window else float(value[0]),
                "state_value_units": 'critic not trained for this risk allocator' if self.policy.cfg.covariance_window else "discounted shaped reward; not a return forecast"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", required=True)
    parser.add_argument("--request", required=True, help="JSON with tickers and observation (market, portfolio, risk)")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if Path(args.output).exists():
        raise FileExistsError("Choose a fresh output path")
    validate_package(args.package)
    cfg = ModelConfig(**json.loads((Path(args.package) / "model_config.json").read_text()))
    shared = SharedBackbone.load(args.device, dtype=cfg.dtype)
    finance = FinanceSpecialist(args.package, shared)
    router = DomainRouter(shared)
    finance.register(router)
    result = router.predict("finance", json.loads(Path(args.request).read_text()))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
