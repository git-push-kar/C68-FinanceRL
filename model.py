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
        super().__init__()
        # A5000 Ampere (sm_86, 24GB): load in bf16 for 2x VRAM/throughput vs fp32; fallback to fp32 on CPU
        dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
        requested = dtype_map.get(getattr(cfg, "dtype", "bf16"), torch.bfloat16)
        # Only use bf16/fp16 on CUDA; CPU stays fp32
        load_dtype = requested if torch.cuda.is_available() else torch.float32
        # InternLM2.5 trust_remote_code model supports torch_dtype arg
        try:
            base = AutoModel.from_pretrained(cfg.backbone_id, trust_remote_code=cfg.trust_remote_code, torch_dtype=load_dtype)
        except TypeError:
            base = AutoModel.from_pretrained(cfg.backbone_id, trust_remote_code=cfg.trust_remote_code)
            if load_dtype != torch.float32:
                base = base.to(dtype=load_dtype)
        for parameter in base.parameters(): parameter.requires_grad = False
        # Auto-correct target_modules for InternLM2 (uses fused wqkv/wo, not q_proj/v_proj)
        # If cfg still has Llama-style names, remap to avoid ValueError: Target modules {'q_proj','v_proj'} not found
        target = list(cfg.target_modules)
        if any(t in ("q_proj", "k_proj", "v_proj", "o_proj") for t in target):
            # detect actual leaf names in base model
            leaf_names = {n.split(".")[-1] for n, m in base.named_modules() if hasattr(m, "weight") and m.weight.ndim == 2}
            if "wqkv" in leaf_names:
                # fused QKV model: map q/k/v -> wqkv, o -> wo
                remapped = []
                for t in target:
                    if t in ("q_proj", "k_proj", "v_proj", "qkv_proj"):
                        if "wqkv" not in remapped: remapped.append("wqkv")
                    elif t in ("o_proj",):
                        remapped.append("wo")
                    else:
                        remapped.append(t)
                target = remapped
                print(f"[InternLMFinancePolicy] Remapped LoRA target_modules {cfg.target_modules} -> {target} (detected wqkv/wo in backbone)")
            elif leaf_names.intersection({"q_proj", "v_proj"}):
                pass  # already compatible
        lora = LoraConfig(r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
                          target_modules=target, bias="none", task_type="FEATURE_EXTRACTION")
        try:
            self.backbone = get_peft_model(base, lora, adapter_name=cfg.adapter_name)
        except ValueError as e:
            if "Target modules" in str(e):
                # last-resort fallback: use whatever Linear leaves exist
                leaf_names = sorted({n.split(".")[-1] for n, m in base.named_modules() if "Linear" in type(m).__name__})
                raise ValueError(f"{e}. Detected Linear leaves in backbone: {leaf_names}. Set ModelConfig.target_modules to one of them, e.g. ['wqkv','wo'] for InternLM2.") from e
            raise
        hidden = getattr(base.config, "hidden_size", cfg.hidden_size); self.encoders = FinanceEncoders(hidden, n_assets)
        # Keep encoders/heads in same dtype as backbone on CUDA for autocast efficiency
        if torch.cuda.is_available() and load_dtype != torch.float32:
            self.encoders = self.encoders.to(dtype=load_dtype)
            # actor/critic stay in load_dtype; PPO loss handles fp32 conversion via autocast
        self.actor, self.critic = nn.Linear(hidden, n_assets), nn.Linear(hidden, 1)
        if torch.cuda.is_available() and load_dtype != torch.float32:
            self.actor = self.actor.to(dtype=load_dtype)
            self.critic = self.critic.to(dtype=load_dtype)
        self.log_std = nn.Parameter(torch.full((n_assets,), -0.5))
    def forward(self, obs):
        # A5000 bf16: encoders/heads are in load_dtype (bf16) but obs from env/tensor_obs is fp32.
        # Cast inputs to encoder dtype for matmul consistency (with or without autocast).
        enc_dtype = next(self.encoders.parameters()).dtype
        market = obs["market"].to(dtype=enc_dtype) if obs["market"].dtype != enc_dtype else obs["market"]
        portfolio = obs["portfolio"].to(dtype=enc_dtype) if obs["portfolio"].dtype != enc_dtype else obs["portfolio"]
        risk = obs["risk"].to(dtype=enc_dtype) if obs["risk"].dtype != enc_dtype else obs["risk"]
        # InternLM2ForCausalLM + transformers 5.x: must disable cache (DynamicCache.from_legacy_cache removed)
        # and request hidden_states; output is CausalLMOutputWithPast (logits + hidden_states), not last_hidden_state
        output = self.backbone(
            inputs_embeds=self.encoders(market, portfolio, risk),
            return_dict=True, use_cache=False, output_hidden_states=True,
        )
        if hasattr(output, "hidden_states") and output.hidden_states is not None:
            state = output.hidden_states[-1][:, -1]
        elif hasattr(output, "last_hidden_state"):
            state = output.last_hidden_state[:, -1]
        else:
            # fallback: last hidden is logits proj input, use logits last dim? should not happen
            raise AttributeError(f"Backbone output has no hidden_states/last_hidden_state: {type(output)} {dir(output)}")
        return self.actor(state), self.critic(state).squeeze(-1)
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
