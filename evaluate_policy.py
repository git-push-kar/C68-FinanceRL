"""Holdout evaluator for trained Finance LoRA + PPO heads vs baselines."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from env import PortfolioEnv
from config import ModelConfig
from model import InternLMFinancePolicy

def load_policy(artifact_dir: Path, device: torch.device):
    cfg_path = artifact_dir / "model_config.json"
    ppo_heads = artifact_dir / "ppo_heads.pt"
    lora_dir = artifact_dir / "finance_lora"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing {cfg_path} — train with train.py first")
    if not lora_dir.exists():
        raise FileNotFoundError(f"Missing {lora_dir}")
    model_cfg = json.loads(cfg_path.read_text())
    # infer n_assets from environment_config if present
    env_cfg_path = artifact_dir / "environment_config.json"
    n_assets = None
    if env_cfg_path.exists():
        env_cfg = json.loads(env_cfg_path.read_text())
        tickers = env_cfg.get("tickers", [])
        n_assets = len(tickers)
    # fallback: read ppo_heads to infer actor shape
    state = torch.load(ppo_heads, map_location="cpu") if ppo_heads.exists() else None
    if n_assets is None and state is not None:
        n_assets = state["actor"].shape[0] if "actor" in state else None
    if n_assets is None:
        raise ValueError("Could not infer n_assets; ensure environment_config.json exists")
    cfg = ModelConfig(**{k: v for k, v in model_cfg.items() if k in ModelConfig.__dataclass_fields__})
    # Force device-appropriate dtype: keep artifact dtype if possible
    policy = InternLMFinancePolicy(cfg, n_assets).to(device)
    # load LoRA weights (PEFT) — handle both finance_lora/adapter_config.json and finance_lora/finance/adapter_config.json
    import os
    adapter_path = lora_dir
    if not (lora_dir / "adapter_config.json").exists():
        # look for nested adapter_name subdir (PEFT save_pretrained with adapter_name creates subdir)
        for cand in [lora_dir / cfg.adapter_name, lora_dir / "finance", lora_dir / "default"]:
            if (cand / "adapter_config.json").exists():
                adapter_path = cand
                break
        # also search recursively one level
        if adapter_path == lora_dir:
            for p in lora_dir.rglob("adapter_config.json"):
                adapter_path = p.parent
                break
    from peft import PeftModel
    try:
        # Use PeftModel.from_pretrained on fresh base to avoid adapter name clash
        from transformers import AutoModel
        base = AutoModel.from_pretrained(cfg.backbone_id, trust_remote_code=cfg.trust_remote_code)
        policy.backbone = PeftModel.from_pretrained(base, str(adapter_path), adapter_name=cfg.adapter_name)
    except Exception as e:
        # fallback: try loading via existing backbone's base_model
        try:
            policy.backbone.load_adapter(str(adapter_path), adapter_name=cfg.adapter_name)
        except Exception:
            raise RuntimeError(f"Failed to load LoRA from {adapter_path} (searched {lora_dir}): {e}") from e
    # ensure whole policy on target device (backbone was loaded on CPU)
    policy = policy.to(device)
    if state is not None:
        policy.encoders.load_state_dict(state["encoders"])
        policy.actor.load_state_dict(state["actor"])
        policy.critic.load_state_dict(state["critic"])
        policy.log_std.data = state["log_std"].to(policy.log_std.device)
    policy.eval()
    return policy, n_assets

def tensor_obs(obs, device):
    return {k: torch.as_tensor(v, device=device).unsqueeze(0) for k, v in obs.items()}

@torch.no_grad()
def run_episode(policy, env, device, use_amp, amp_dtype, deterministic=True):
    obs, _ = env.reset()
    infos = []
    while True:
        obs_t = tensor_obs(obs, device)
        if use_amp:
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                mean, _ = policy(obs_t)
        else:
            mean, _ = policy(obs_t)
        # deterministic: use mean action; stochastic would sample via policy.act
        action = mean.squeeze(0).float().cpu().numpy() if deterministic else policy.act(obs_t)[0].squeeze(0).float().cpu().numpy()
        obs, _, terminated, truncated, info = env.step(action)
        infos.append(info)
        if terminated or truncated:
            break
    return infos

def metrics_from_infos(infos, start_equity=100_000):
    if not infos:
        return {}
    equities = [x["equity"] for x in infos]
    nets = [x["net_return"] for x in infos]
    draws = [x["drawdown"] for x in infos]
    turns = [x["turnover"] for x in infos]
    return {
        "final_equity": float(equities[-1]),
        "peak_equity": float(max(equities)),
        "return_total": float(equities[-1]/start_equity - 1),
        "return_annualized": float(np.mean(nets)*252),
        "mean_net_return": float(np.mean(nets)),
        "mean_gross_return": float(np.mean([x["gross_return"] for x in infos])),
        "mean_turnover": float(np.mean(turns)),
        "mean_trading_cost": float(np.mean([x["trading_cost"] for x in infos])),
        "max_drawdown": float(max(draws)),
        "volatility_daily": float(np.std(nets)),
        "sharpe_proxy": float(np.mean(nets)/(np.std(nets)+1e-8) * np.sqrt(252)),
        "steps": len(infos),
    }

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Evaluate trained Finance LoRA on holdout vs equal_weight/baseline")
    p.add_argument("--data", required=True)
    p.add_argument("--artifacts", required=True, help="train.py --output dir (contains finance_lora + ppo_heads.pt)")
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--window", type=int, default=30)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--deterministic", action="store_true", default=True)
    p.add_argument("--stochastic", dest="deterministic", action="store_false")
    p.add_argument("--dtype", default=None, help="override dtype (bf16/fp32); default uses artifact")
    p.add_argument("--cost-bps", type=float, default=None, help="override transaction cost; default from artifact")
    args = p.parse_args()

    device = torch.device(args.device)
    artifact_dir = Path(args.artifacts)
    env_cfg_path = artifact_dir / "environment_config.json"
    # load env tickers to ensure same assets
    start_equity = 100_000.0

    policy, n_assets = load_policy(artifact_dir, device)
    # dtype/amp for eval
    use_amp = device.type == "cuda" and getattr(policy, "backbone", None) is not None
    # infer backbone dtype
    try:
        bdtype = next(policy.backbone.parameters()).dtype
        amp_dtype = bdtype if bdtype in (torch.bfloat16, torch.float16) else torch.bfloat16
        if args.dtype == "fp32":
            use_amp = False
    except Exception:
        amp_dtype = torch.bfloat16
        if args.dtype == "fp32":
            use_amp = False

    # Baseline envs share same window/filter
    # 1) Learned policy
    env_lo = PortfolioEnv(args.data, window=args.window, start_date=args.start, end_date=args.end)
    infos_lo = run_episode(policy, env_lo, device, use_amp if device.type=="cuda" else False, amp_dtype, deterministic=args.deterministic)
    m_lo = metrics_from_infos(infos_lo, start_equity)
    env_lo.close()

    # 2) Equal-weight baseline (same period)
    env_eq = PortfolioEnv(args.data, window=args.window, start_date=args.start, end_date=args.end)
    obs, _ = env_eq.reset()
    infos_eq = []
    while True:
        obs, _, terminated, truncated, info = env_eq.step(np.zeros(env_eq.n_assets, dtype=np.float32))
        infos_eq.append(info)
        if terminated or truncated:
            break
    m_eq = metrics_from_infos(infos_eq, start_equity)
    env_eq.close()

    # 3) Buy-and-hold equal -> already same as equal_weight daily rebalanced; also compute buy-hold without rebalance proxy via equal_weight
    print(json.dumps({"holdout": f"{args.start}→{args.end}", "n_assets": n_assets, "window": args.window}, indent=2))
    print(json.dumps({"learned_policy": m_lo}, indent=2))
    print(json.dumps({"baseline_equal_weight": m_eq}, indent=2))
    # summary comparison
    if m_eq["final_equity"] > 0:
        alpha = (m_lo["final_equity"] - m_eq["final_equity"]) / m_eq["final_equity"]
        print(json.dumps({"alpha_vs_equal_weight": float(alpha), "sharpe_learned": m_lo["sharpe_proxy"], "sharpe_eq": m_eq["sharpe_proxy"], "dd_learned": m_lo["max_drawdown"], "dd_eq": m_eq["max_drawdown"]}, indent=2))

    # Also dump to artifact dir for record
    out = artifact_dir / f"eval_{args.start}_{args.end}.json"
    out.write_text(json.dumps({"holdout": f"{args.start}→{args.end}", "learned_policy": m_lo, "baseline_equal_weight": m_eq}, indent=2))
    print(f"Saved eval to {out}")
