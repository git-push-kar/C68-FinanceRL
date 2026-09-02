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
    # InternLM2 uses fused wqkv + wo (not q_proj/v_proj like Llama); w1/w2/w3 are FFN
    target_modules: List[str] = field(default_factory=lambda: ["wqkv", "wo"])
    adapter_name: str = "finance"
    # A5000 (Ampere) prefers bf16; options: bf16, fp16, fp32
    dtype: str = "bf16"


@dataclass
class PPOConfig:
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    rollout_steps: int = 256  # prod on A5000: 512 (CLI --rollout-steps 512)
    update_epochs: int = 4
    minibatch_size: int = 64  # prod on A5000: 128 (CLI --minibatch-size 128); 24GB can use 256
    # A5000 (sm_86 24GB) prod recipe: 75k-100k timesteps (≈70-94 train-split episodes) + rollout 512/minibatch 128/bf16


def serializable(config):
    return asdict(config)
