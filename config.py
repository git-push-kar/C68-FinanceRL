from dataclasses import asdict, dataclass, field
from typing import List


@dataclass
class ModelConfig:
    backbone_id: str = "internlm/internlm2_5-1_8b-chat"
    trust_remote_code: bool = True
    hidden_size: int = 2048
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    adapter_name: str = "finance"


@dataclass
class PPOConfig:
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    rollout_steps: int = 256
    update_epochs: int = 4
    minibatch_size: int = 64


def serializable(config):
    return asdict(config)
