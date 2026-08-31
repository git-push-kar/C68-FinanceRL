"""Frozen shared InternLM plus a portable named Finance LoRA and PPO heads."""
import json
from pathlib import Path
import torch
from torch import nn
from peft import LoraConfig, get_peft_model
from transformers import AutoModel


class FinanceEncoders(nn.Module):
    def __init__(self, hidden_size, n_assets):
        super().__init__()
        self.market = nn.Sequential(nn.Linear(5, hidden_size), nn.GELU(), nn.Linear(hidden_size, hidden_size))
        self.portfolio = nn.Sequential(nn.LayerNorm(n_assets + 1), nn.Linear(n_assets + 1, hidden_size), nn.GELU())
        self.risk = nn.Sequential(nn.LayerNorm(3), nn.Linear(3, hidden_size), nn.GELU())
    def forward(self, market, portfolio, risk):
        return torch.stack((self.market(market).mean((1, 2)), self.portfolio(portfolio), self.risk(risk)), dim=1)


class InternLMFinancePolicy(nn.Module):
    def __init__(self, cfg, n_assets):
        super().__init__(); base = AutoModel.from_pretrained(cfg.backbone_id, trust_remote_code=cfg.trust_remote_code)
        for parameter in base.parameters(): parameter.requires_grad = False
        lora = LoraConfig(r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
                          target_modules=cfg.target_modules, bias="none", task_type="FEATURE_EXTRACTION")
        self.backbone = get_peft_model(base, lora, adapter_name=cfg.adapter_name)
        hidden = getattr(base.config, "hidden_size", cfg.hidden_size); self.encoders = FinanceEncoders(hidden, n_assets)
        self.actor, self.critic = nn.Linear(hidden, n_assets), nn.Linear(hidden, 1)
        self.log_std = nn.Parameter(torch.full((n_assets,), -0.5))
    def forward(self, obs):
        output = self.backbone(inputs_embeds=self.encoders(obs["market"], obs["portfolio"], obs["risk"]), return_dict=True)
        state = output.last_hidden_state[:, -1]; return self.actor(state), self.critic(state).squeeze(-1)
    def _dist(self, mean): return torch.distributions.Normal(mean, self.log_std.exp().expand_as(mean))
    def act(self, obs):
        mean, value = self(obs); dist = self._dist(mean); action = dist.sample()
        return action, dist.log_prob(action).sum(-1), value
    def evaluate_actions(self, obs, actions):
        mean, value = self(obs); dist = self._dist(mean)
        return dist.log_prob(actions).sum(-1), value, dist.entropy().sum(-1)
    def save_artifacts(self, output_dir, model_config, ppo_config, reward_config, tickers):
        output = Path(output_dir); output.mkdir(parents=True, exist_ok=True); self.backbone.save_pretrained(output/"finance_lora")
        torch.save({"encoders":self.encoders.state_dict(), "actor":self.actor.state_dict(), "critic":self.critic.state_dict(), "log_std":self.log_std.detach().cpu()}, output/"ppo_heads.pt")
        (output/"model_config.json").write_text(json.dumps(model_config, indent=2)); (output/"ppo_config.json").write_text(json.dumps(ppo_config, indent=2))
        (output/"environment_config.json").write_text(json.dumps({"reward":reward_config, "tickers":tickers}, indent=2))
