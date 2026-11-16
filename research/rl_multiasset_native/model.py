"""Finance tokens and PPO heads on the project's shared InternVL3-2B."""
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import torch
from torch import nn

from config import ARTIFACT_VERSION, BACKBONE_ID, BACKBONE_REVISION, LORA_TARGETS, OBSERVATION_SCHEMA, ModelConfig, lora_targets
from shared_backbone import SharedBackbone


class FinanceEncoders(nn.Module):
    def __init__(self, hidden_size, n_assets, width, *, use_asset_ids=True, normalize_market=False, covariance_context=False):
        super().__init__()
        self.n_assets = n_assets
        self.temporal = nn.GRU(5, width, batch_first=True)
        self.market_projection = nn.Sequential(nn.Linear(width, hidden_size), nn.LayerNorm(hidden_size))
        self.asset_ids = nn.Embedding(n_assets, hidden_size)
        self.use_asset_ids = use_asset_ids
        self.normalize_market = normalize_market
        self.covariance_context = covariance_context
        if covariance_context:
            self.covariance_projection = nn.Sequential(nn.Linear(n_assets+2,width),nn.GELU(),
                nn.Linear(width,hidden_size),nn.LayerNorm(hidden_size))
        self.asset_ids.weight.requires_grad_(use_asset_ids)
        self.portfolio = nn.Sequential(nn.Linear(n_assets + 1, width), nn.GELU(), nn.Linear(width, hidden_size))
        self.risk = nn.Sequential(nn.Linear(3, width), nn.GELU(), nn.Linear(width, hidden_size))
        self.holding = nn.Linear(1, hidden_size)
        self.query = nn.Parameter(torch.randn(hidden_size) * 0.02)
        self.value_query = nn.Parameter(torch.randn(hidden_size) * 0.02)

    def forward(self, market, portfolio, risk, covariance=None):
        batch, window, assets, features = market.shape
        if assets != self.n_assets or features != 5:
            raise ValueError("Market must have shape [batch, window, trained assets, 5]")
        if self.normalize_market:
            # Each feature uses only the known cross section at that historical
            # timestep. Keep small price-return features visible beside volume.
            market = ((market - market.mean(2, keepdim=True)) /
                      market.std(2, unbiased=False, keepdim=True).clamp_min(1e-6)).clamp(-5, 5)
        histories = market.permute(0, 2, 1, 3).reshape(batch * assets, window, 5)
        _, hidden = self.temporal(histories)
        market_tokens = self.market_projection(hidden[-1].reshape(batch, assets, -1))
        if self.covariance_context:
            if covariance is None or covariance.shape != (batch,assets,assets):
                raise ValueError('Covariance-conditioned policy requires its ordered causal covariance matrix')
            variance=covariance.diagonal(dim1=-2,dim2=-1).clamp_min(1e-12)
            vol=variance.sqrt()
            corr=(covariance/(vol.unsqueeze(-1)*vol.unsqueeze(-2))).clamp(-1,1)
            log_vol=vol.log();log_vol=log_vol-log_vol.mean(-1,keepdim=True)
            # Covariance with the known uniform portfolio, relative to its
            # variance; complements relative per-stock volatility/correlation.
            beta=covariance.mean(-1)/covariance.mean((-1,-2),keepdim=False).clamp_min(1e-12).unsqueeze(-1)
            context=torch.cat([corr,log_vol.unsqueeze(-1),beta.unsqueeze(-1)],dim=-1)
            market_tokens=market_tokens+self.covariance_projection(context)
        if self.use_asset_ids:
            market_tokens = market_tokens + self.asset_ids.weight.unsqueeze(0)
        # Queries FOLLOW all market tokens: causal attention sees every stock.
        queries = market_tokens + self.query + self.holding(portfolio[:, :assets].unsqueeze(-1))
        context = torch.stack((self.risk(risk), self.portfolio(portfolio)), dim=1)
        value = self.value_query.view(1, 1, -1).expand(batch, 1, -1)
        return torch.cat((context, market_tokens, queries, value), dim=1)


class InternVLFinancePolicy(nn.Module):
    def __init__(self, cfg, tickers, shared=None, *, initialize_adapter=True, device="cpu"):
        super().__init__()
        self.cfg = cfg
        self.tickers = list(tickers)
        if not self.tickers or len(set(self.tickers)) != len(self.tickers):
            raise ValueError("Tickers must be nonempty and unique; their order is part of the package")
        self.shared = shared if shared is not None else SharedBackbone.load(device, cfg.dtype)
        if initialize_adapter:
            self.shared.add_finance_adapter(cfg)
        elif "finance" not in self.shared.adapter_names:
            raise ValueError("Load the finance adapter before its heads")
        hidden, assets = self.shared.hidden_size, len(self.tickers)
        self.encoders = FinanceEncoders(hidden, assets, cfg.encoder_width,
                                       use_asset_ids=cfg.use_asset_ids, normalize_market=cfg.normalize_market,
                                       covariance_context=bool(cfg.covariance_window))
        self.actor = nn.Linear(hidden, 1)
        self.critic = nn.Linear(hidden, 1)
        with torch.no_grad():
            self.actor.weight.mul_(cfg.actor_init_scale)
            self.actor.bias.mul_(cfg.actor_init_scale)
        self.log_std = nn.Parameter(torch.full((assets,), float(cfg.initial_log_std)))
        # Do not recursively move the resident full backbone: some inactive
        # components may intentionally remain on CPU with their source weights.
        for module in (self.encoders, self.actor, self.critic):
            module.to(self.shared.compute_device)
        self.log_std.data = self.log_std.data.to(self.shared.compute_device)
        if cfg.lora_only:
            for module in (self.encoders,self.actor,self.critic):module.requires_grad_(False)
            self.log_std.requires_grad_(False)
        self.train()

    def train(self, mode=True):
        super().train(mode)
        # eval disables dropout, not gradients; PPO needs consistent likelihoods.
        self.shared.model.eval()
        return self

    def forward(self, obs):
        # Rollouts use batch size one. Keep the same kernel shapes during PPO
        # reevaluation: bf16 batch kernels can otherwise change the means enough
        # to trigger a false KL stop before the first optimizer step.
        batch = obs["market"].shape[0]
        if batch > 1:
            outputs = [self._forward_native({key: value[i:i + 1] for key, value in obs.items()})
                       for i in range(batch)]
            return torch.cat([out[0] for out in outputs]), torch.cat([out[1] for out in outputs])
        return self._forward_native(obs)

    def _forward_native(self, obs):
        if obs["market"].shape[1] != self.cfg.window:
            raise ValueError("Observation window differs from trained finance package")
        device_type = obs["market"].device.type
        with torch.autocast(device_type=device_type, enabled=False):
            portfolio,risk=obs['portfolio'].float(),obs['risk'].float()
            if self.cfg.state_invariant:
                portfolio=torch.zeros_like(portfolio)
                portfolio[:,:-1]=1/len(self.tickers)
                risk=torch.zeros_like(risk)
            tokens = self.encoders(obs["market"].float(), portfolio, risk,
                obs['covariance'].float() if self.cfg.covariance_window else None)
        states = self.shared.finance_hidden_states(tokens)
        assets = len(self.tickers)
        with torch.autocast(device_type=device_type, enabled=False):
            mean = self.actor(states[:, 2 + assets:2 + 2 * assets].float()).squeeze(-1) * self.cfg.score_scale
            value = self.critic(states[:, -1].float()).squeeze(-1)
        return mean, value

    def _dist(self, mean):
        return torch.distributions.Normal(mean.float(), self.log_std.clamp(-5, 1).exp().expand_as(mean))

    def act(self, obs):
        mean, value = self(obs)
        dist = self._dist(mean)
        action = dist.sample()
        return action, dist.log_prob(action).sum(-1), value

    def evaluate_actions(self, obs, actions):
        mean, value = self(obs)
        dist = self._dist(mean)
        return dist.log_prob(actions.float()).sum(-1), value, dist.entropy().sum(-1)

    def head_state(self):
        return {"encoders": self.encoders.state_dict(), "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(), "log_std": self.log_std.detach().cpu()}

    def save_artifacts(self, output_dir, ppo_config, reward_config, run_metadata=None):
        output = Path(output_dir)
        if (output / "manifest.json").exists():
            raise FileExistsError("Use a new package directory; never overwrite an existing manifest")
        output.mkdir(parents=True, exist_ok=True)
        # Numeric input encoders are saved below; vocabulary embeddings stay frozen.
        self.shared.model.save_pretrained(output / "finance_lora", selected_adapters=["finance"],
                                         save_embedding_layers=False)
        torch.save(self.head_state(), output / "finance_heads.pt")
        write_json(output / "model_config.json", asdict(self.cfg))
        write_json(output / "ppo_config.json", ppo_config)
        write_json(output / "environment_config.json", {"reward": reward_config, "tickers": self.tickers,
                                                        "window": self.cfg.window, "simulator_version": 3 if self.cfg.covariance_window else 2,
                                                        "covariance_window": self.cfg.covariance_window})
        files = {p.relative_to(output).as_posix(): digest(p) for p in output.rglob("*") if p.is_file()}
        write_json(output / "manifest.json", {
            "format_version": 3 if self.cfg.covariance_window else ARTIFACT_VERSION, "domain": "finance", "adapter_name": "finance",
            "base_model_id": BACKBONE_ID, "base_revision": BACKBONE_REVISION,
            "adapter_scope": "full_internvl_root", "target_modules": lora_targets(self.cfg),
            "numeric_entry_point": "language_model.model", "hidden_size": self.shared.hidden_size,
            "observation_schema": observation_schema(self.cfg),
            "architecture_test_only": self.shared.architecture_test,
            "tickers": self.tickers, "window": self.cfg.window,
            "run": run_metadata or {}, "sha256": files,
        })

    @classmethod
    def load_artifacts(cls, artifact_dir, shared=None, device="cpu", *, allow_architecture_test=False, for_training=False):
        path = Path(artifact_dir)
        manifest = validate_package(path, allow_architecture_test=allow_architecture_test)
        cfg = ModelConfig(**json.loads((path / "model_config.json").read_text(encoding="utf-8")))
        runtime = shared if shared is not None else SharedBackbone.load(device, cfg.dtype)
        if runtime.architecture_test != manifest["architecture_test_only"]:
            raise ValueError("Cannot mix architecture-test packages and pretrained models")
        if runtime.hidden_size != manifest["hidden_size"]:
            raise ValueError("Finance hidden size differs from resident backbone")
        runtime.load_adapter(path / "finance_lora" / "finance", "finance", trainable=for_training)
        policy = cls(cfg, manifest["tickers"], runtime, initialize_adapter=False)
        state = torch.load(path / "finance_heads.pt", map_location="cpu", weights_only=True)
        for key in ("encoders", "actor", "critic"):
            getattr(policy, key).load_state_dict(state[key], strict=True)
        with torch.no_grad():
            policy.log_std.copy_(state["log_std"].to(policy.log_std.device))
        if for_training:
            return policy.train()
        policy.requires_grad_(False)
        return policy.eval()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, obj):
    path=Path(path)
    temporary=path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(obj, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def observation_schema(cfg):
    if not cfg.covariance_window:
        return OBSERVATION_SCHEMA
    schema={**OBSERVATION_SCHEMA,'version':'finance_covariance_v3',
        'covariance':{'estimator':'LedoitWolf','returns':cfg.covariance_window,
                      'ordered_axes':'package ticker order on both axes','availability':'through decision close only'}}
    if cfg.state_invariant:
        schema['policy_state']='neutral internal holdings/risk: expert target is state independent; execution still uses actual holdings'
    return schema


def validate_package(path, *, allow_architecture_test=False):
    path = Path(path)
    if not (path / "manifest.json").is_file():
        raise ValueError("Missing v2 manifest. Legacy InternLM runs cannot load into InternVL3; retrain finance.")
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    cfg = ModelConfig(**json.loads((path / "model_config.json").read_text(encoding="utf-8")))
    expected = {"format_version": 3 if cfg.covariance_window else ARTIFACT_VERSION, "base_model_id": BACKBONE_ID,
                "base_revision": BACKBONE_REVISION, "domain": "finance", "adapter_name": "finance",
                "adapter_scope": "full_internvl_root", "target_modules": lora_targets(cfg),
                "numeric_entry_point": "language_model.model", "observation_schema": observation_schema(cfg)}
    if any(manifest.get(k) != v for k, v in expected.items()):
        raise ValueError("Finance package does not match shared-backbone contract")
    if manifest.get("architecture_test_only") and not allow_architecture_test:
        raise ValueError("Random tiny architecture-test artifacts are not trained finance adapters")
    required = {"finance_heads.pt", "model_config.json", "ppo_config.json", "environment_config.json",
                "finance_lora/finance/adapter_config.json", "finance_lora/finance/adapter_model.safetensors"}
    if not required.issubset(manifest.get("sha256", {})):
        raise ValueError("Finance package is incomplete")
    for relative, expected_hash in manifest["sha256"].items():
        file = (path / relative).resolve()
        if not file.is_relative_to(path.resolve()) or not file.is_file() or digest(file) != expected_hash:
            raise ValueError(f"Missing or changed finance file: {relative}")
    cfg = ModelConfig(**json.loads((path / "model_config.json").read_text(encoding="utf-8")))
    env = json.loads((path / "environment_config.json").read_text(encoding="utf-8"))
    if env["tickers"] != manifest["tickers"] or env["window"] != manifest["window"] or cfg.window != manifest["window"]:
        raise ValueError("Inconsistent ticker/window metadata")
    if env.get('covariance_window',0)!=cfg.covariance_window:
        raise ValueError('Inconsistent covariance context metadata')
    return manifest
