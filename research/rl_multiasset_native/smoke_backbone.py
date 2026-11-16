"""Test actual upstream InternVL code at tiny dimensions or full pretrained size.

Tiny mode downloads architecture/config only; it NEVER produces a trained adapter.
Full mode downloads pinned pretrained weights and is intended for the college GPU.
"""
import argparse
from pathlib import Path
import tempfile

import torch
from transformers import AutoConfig, AutoModel
from peft import LoraConfig

from config import BACKBONE_ID, BACKBONE_REVISION, LORA_TARGETS, ModelConfig, PPOConfig, serializable
from env import FinanceRewardConfig
from model import InternVLFinancePolicy, write_json
from shared_backbone import DomainRouter, SharedBackbone


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tiny", action="store_true")
    parser.add_argument("--source", help="Optional local official-code directory for offline tiny tests")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    parser.add_argument("--output", required=True, help="Fresh verification report JSON")
    args = parser.parse_args()
    if Path(args.output).exists():
        raise FileExistsError("Choose a new verification report path")
    torch.set_num_threads(1)
    device = torch.device(args.device)

    def make_shared():
        if not args.tiny:
            return SharedBackbone.load(device, args.dtype)
        source = args.source or BACKBONE_ID
        config = AutoConfig.from_pretrained(source, trust_remote_code=True, revision=BACKBONE_REVISION)
        config.force_image_size = 28
        config.vision_config.image_size = 28
        config.vision_config.hidden_size = 32
        config.vision_config.intermediate_size = 64
        config.vision_config.num_hidden_layers = 2
        config.vision_config.num_attention_heads = 4
        config.vision_config.drop_path_rate = 0
        config.llm_config.hidden_size = 32
        config.llm_config.intermediate_size = 64
        config.llm_config.num_hidden_layers = 2
        config.llm_config.num_attention_heads = 4
        config.llm_config.num_key_value_heads = 2
        config.llm_config.vocab_size = 64
        config.llm_config.rope_scaling = None
        config.llm_config.use_sliding_window = False
        config.llm_config.sliding_window = None
        config.hidden_size = 32
        with torch.random.fork_rng():
            torch.manual_seed(123)
            base = AutoModel.from_config(config, trust_remote_code=True, use_flash_attn=False,
                                         torch_dtype=torch.float32).to(device)
        return SharedBackbone(base, architecture_test=True)

    shared = make_shared()
    policy = InternVLFinancePolicy(ModelConfig(dtype="fp32" if args.tiny else args.dtype),
                                    [f"ASSET_{i}" for i in range(49)], shared)
    obs = {"market": torch.randn(1, 30, 49, 5, device=device),
           "portfolio": torch.full((1, 50), .02, device=device),
           "risk": torch.zeros(1, 3, device=device)}
    action, old_logp, value = policy.act(obs)
    new_logp, _, _ = policy.evaluate_actions(obs, action)
    torch.testing.assert_close(old_logp, new_logp)
    (policy(obs)[0].square().mean() + value.square().mean()).backward()
    frozen_with_grad = [n for n, p in shared.named_parameters() if "lora_" not in n and p.grad is not None]
    assert not frozen_with_grad, frozen_with_grad
    assert any(p.grad is not None and torch.count_nonzero(p.grad) for n, p in shared.named_parameters() if "lora_" in n)
    four_adapter_test = False
    if args.tiny:
        # Synthetic test adapters validate isolation, not domain accuracy.
        for name in ("deepfake_test", "anomaly_test", "reasoning_test"):
            shared.model.add_adapter(name, LoraConfig(r=2, target_modules=LORA_TARGETS))
        with torch.no_grad():
            for name, parameter in shared.named_parameters():
                if "lora_B" in name and "_test" in name:
                    parameter.normal_(0, .03)
        router = DomainRouter(shared)
        probe = torch.randn(1, 7, shared.hidden_size, device=device)
        def handler(model, payload):
            base = model.get_base_model()
            payload = payload.to(base.language_model.get_input_embeddings().weight.dtype)
            return base.language_model.model(inputs_embeds=payload, use_cache=False).last_hidden_state.clone()
        for name in ("finance", "deepfake_test", "anomaly_test", "reasoning_test"):
            router.register(name, name, handler)
        reference = router.predict("finance", probe)
        base_identity = id(shared.base)
        for name in ("deepfake_test", "anomaly_test", "reasoning_test"):
            assert not torch.allclose(reference, router.predict(name, probe))
            assert shared.model.active_adapter == "finance"
            assert id(shared.base) == base_identity
        torch.testing.assert_close(reference, router.predict("finance", probe))
        four_adapter_test = True
    # Verify the SAME model still exposes and runs its vision path.
    image_size = shared.base.config.force_image_size
    with torch.no_grad():
        image_dtype = next(shared.base.vision_model.parameters()).dtype
        visual = shared.base.extract_feature(torch.randn(1, 3, image_size, image_size, device=device, dtype=image_dtype))
        expected = policy(obs)[0].cpu()
    # Reload tiny model only: loading a second full 2B base defeats the memory
    # objective. Full package reload must also pass after the college GPU run.
    roundtrip = False
    if args.tiny:
        with tempfile.TemporaryDirectory() as folder:
            policy.save_artifacts(folder, serializable(PPOConfig()), serializable(FinanceRewardConfig()))
            restored = InternVLFinancePolicy.load_artifacts(folder, make_shared(), allow_architecture_test=True)
            with torch.no_grad():
                torch.testing.assert_close(expected, restored(obs)[0].cpu(), rtol=1e-5, atol=1e-6)
            roundtrip = True
    result = {"mode": "tiny_random_official_architecture" if args.tiny else "full_pretrained_smoke",
              "base_model_id": BACKBONE_ID, "base_revision": BACKBONE_REVISION,
              "dtype": "fp32" if args.tiny else args.dtype,
              "official_model_class": type(shared.base).__name__, "hidden_size": shared.hidden_size,
              "action_shape": list(action.shape), "vision_feature_shape": list(visual.shape),
              "likelihood_consistent": True, "lora_gradients": True, "frozen_base_gradients": False,
              "tiny_save_load_passed": roundtrip, "trained_finance_adapter": False}
    result["four_synthetic_adapter_isolation_on_tiny"] = four_adapter_test
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, result)
    print(result)


if __name__ == "__main__":
    main()
