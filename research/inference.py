"""Load finance beside other specialists on ONE resident InternVL3 model."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from allocation import allocation_weights
from config import BACKBONE_ID, BACKBONE_REVISION, ModelConfig
from model import InternVLFinancePolicy, validate_package, write_json
from shared_backbone import DomainRouter, SharedBackbone


class FinanceSpecialist:
    def __init__(self, package_dir, shared):
        self.policy = InternVLFinancePolicy.load_artifacts(package_dir, shared=shared)
        self.environment = json.loads((Path(package_dir) / "environment_config.json").read_text())

    def register(self, router):
        if router.shared is not self.policy.shared:
            raise ValueError("Finance and router must reference the same resident backbone")
        router.register("finance", "finance", self.predict)

    def predict(self, model, payload):
        if model is not self.policy.shared.model:
            raise ValueError("Finance handler received a different model instance")
        if payload.get("tickers") != self.policy.tickers:
            raise ValueError("Supply the exact trained ticker order with every finance request")
        obs = payload["observation"]
        assets, window = len(self.policy.tickers), self.policy.cfg.window
        shapes = {"market": (window, assets, 5), "portfolio": (assets + 1,), "risk": (3,)}
        device = next(self.policy.parameters()).device
        tensors = {}
        for key, shape in shapes.items():
            array = np.asarray(obs[key], dtype=np.float32)
            if array.shape != shape or not np.isfinite(array).all():
                raise ValueError(f"Observation {key} must have finite values and shape {shape}")
            tensors[key] = torch.as_tensor(array, device=device).unsqueeze(0)
        holdings = tensors["portfolio"][0]
        if (holdings < 0).any() or not torch.isclose(holdings.sum(), torch.tensor(1., device=device), atol=1e-5):
            raise ValueError("Portfolio weights, including cash, must be nonnegative and sum to one")
        with torch.inference_mode():
            scores, value = self.policy(tensors)
        weights = allocation_weights(scores[0].cpu().numpy(), self.environment["reward"]["max_weight"],
                                     holdings[:-1].cpu().numpy(), self.environment["reward"].get("rebalance_rate", 1.0))
        return {"domain": "finance", "base_model_id": BACKBONE_ID, "base_revision": BACKBONE_REVISION,
                "tickers": self.policy.tickers, "weights": weights.tolist(), "cash_weight": 0.0,
                "state_value": float(value[0]), "state_value_units": "discounted shaped reward; not a return forecast"}


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
