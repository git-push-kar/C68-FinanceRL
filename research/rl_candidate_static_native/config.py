"""InternVL3-2B is the fixed shared foundation for every domain."""
from dataclasses import asdict, dataclass

BACKBONE_ID = "OpenGVLab/InternVL3-2B"
BACKBONE_REVISION = "899155015275a9b7338c7f4677e19c784e0e5a21"
LORA_TARGETS = r"language_model\.model\.layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj|o_proj)"
LORA_ATTENTION_MLP_TARGETS = r"language_model\.model\.layers\.\d+\.(self_attn\.(q_proj|k_proj|v_proj|o_proj)|mlp\.(gate_proj|up_proj|down_proj))"
ARTIFACT_VERSION = 2
OBSERVATION_SCHEMA = {
    "version": "finance_prices_v2",
    "market_features": ["simple_return", "log_return", "window_rolling_volatility", "moving_average_gap", "volume_zscore"],
    "portfolio": "ordered_asset_weights_then_cash",
    "risk": ["recent_return_volatility", "drawdown", "equity_over_initial_minus_one"],
}


@dataclass
class ModelConfig:
    backbone_id: str = BACKBONE_ID
    backbone_revision: str = BACKBONE_REVISION
    adapter_name: str = "finance"
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    encoder_width: int = 64
    window: int = 30
    dtype: str = "bf16"
    score_scale: float = 1.0
    actor_init_scale: float = 1.0
    initial_log_std: float = -1.5
    use_asset_ids: bool = True
    normalize_market: bool = False
    covariance_window: int = 0
    state_invariant: bool = False
    lora_only: bool = False
    lora_scope: str = 'attention'

    def __post_init__(self):
        if (self.backbone_id, self.backbone_revision) != (BACKBONE_ID, BACKBONE_REVISION):
            raise ValueError("Finance requires pinned OpenGVLab/InternVL3-2B; legacy InternLM artifacts must be retrained.")
        if self.adapter_name != "finance":
            raise ValueError("This package reserves adapter name 'finance'.")
        if self.dtype not in {"fp32", "bf16", "fp16"}:
            raise ValueError("dtype must be fp32, bf16 or fp16")
        if min(self.lora_r, self.lora_alpha, self.encoder_width, self.window) < 1:
            raise ValueError("Model dimensions must be positive")
        if self.lora_dropout != 0:
            raise ValueError("Finance PPO requires zero LoRA dropout for consistent action likelihoods.")
        if self.covariance_window not in (0, 252):
            raise ValueError('Covariance context must be disabled or use 252 known returns')
        if not isinstance(self.state_invariant,bool) or not isinstance(self.lora_only,bool):
            raise ValueError('Adaptation flags must be boolean')
        if (self.state_invariant or self.lora_only) and not self.covariance_window:
            raise ValueError('These adaptation flags require the covariance schema')
        if self.lora_scope not in ('attention','attention_mlp','attention_readout'):
            raise ValueError('Unknown finance LoRA scope')
        import math
        if not all(math.isfinite(x) for x in (self.score_scale, self.actor_init_scale, self.initial_log_std)):
            raise ValueError("Policy scale settings must be finite")
        if self.score_scale <= 0 or not 0 < self.actor_init_scale <= 1 or not -5 <= self.initial_log_std <= 1:
            raise ValueError("Invalid policy scale or initial exploration setting")


@dataclass
class PPOConfig:
    learning_rate: float = 1e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.001
    max_grad_norm: float = 0.5
    rollout_steps: int = 256
    update_epochs: int = 4
    minibatch_size: int = 16
    target_kl: float = 0.02


def serializable(config):
    return asdict(config)


def lora_targets(config):
    if config.lora_scope=='attention_readout':return '('+LORA_TARGETS+r'|language_model\.lm_head)'
    return LORA_ATTENTION_MLP_TARGETS if config.lora_scope=='attention_mlp' else LORA_TARGETS
