"""Standalone historical Finance LoRA training script; never connects to a broker.
A5000 (Ampere, 24GB, sm_86) optimized: bf16 autocast, TF32, torch.compile, larger batches.
"""
import argparse
from pathlib import Path
import numpy as np
import torch
from config import ModelConfig, PPOConfig, serializable
from env import FinanceRewardConfig, PortfolioEnv
from model import InternLMFinancePolicy
from ppo import compute_gae, ppo_update


def tensor_obs(obs, device): return {k: torch.as_tensor(v, device=device).unsqueeze(0) for k, v in obs.items()}

def _metrics(infos):
    if not infos: return {}
    nets = [x["net_return"] for x in infos]; draws = [x["drawdown"] for x in infos]; turns = [x["turnover"] for x in infos]
    import numpy as _np
    return {"mean_net": float(_np.mean(nets)), "max_dd": float(max(draws)), "mean_turnover": float(_np.mean(turns))}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--timesteps", type=int, default=25_000)
    parser.add_argument("--window", type=int, default=30)
    parser.add_argument("--output", default="artifacts/finance")
    parser.add_argument("--backbone", default=ModelConfig.backbone_id)
    parser.add_argument("--cost-bps", type=float, default=10.0)
    parser.add_argument("--max-weight", type=float, default=0.30)
    parser.add_argument("--train-end")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    # A5000 flags
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16", help="Ampere bf16 is fastest/stable; fp16 needs scaler")
    parser.add_argument("--amp", action="store_true", default=True, help="Enable autocast on CUDA (default on for cuda)")
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.add_argument("--compile", action="store_true", help="torch.compile encoders+heads (PyTorch 2.4+)")
    parser.add_argument("--minibatch-size", type=int, default=None, help="Override PPO minibatch (A5000: 128 or 256)")
    parser.add_argument("--rollout-steps", type=int, default=None, help="Override PPO rollout_steps (A5000: 512)")
    parser.add_argument("--eval-interval", type=int, default=5000, help="Holdout eval interval steps (0=off)")
    parser.add_argument("--eval-start", default="2023-01-01", help="Holdout start for periodic eval")
    parser.add_argument("--eval-end", default="2024-12-31", help="Holdout end for periodic eval")
    parser.add_argument("--checkpoint-interval", type=int, default=5000, help="Checkpoint save interval steps (0=off)")
    args = parser.parse_args()

    # A5000 throughput knobs: TF32 for matmuls (Ampere)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    env_cfg = FinanceRewardConfig(transaction_cost_bps=args.cost_bps, max_weight=args.max_weight)
    env = PortfolioEnv(args.data, window=args.window, reward_config=env_cfg, end_date=args.train_end)
    cfg, ppo_cfg = ModelConfig(backbone_id=args.backbone, dtype=args.dtype), PPOConfig()
    if args.minibatch_size: ppo_cfg.minibatch_size = args.minibatch_size
    if args.rollout_steps: ppo_cfg.rollout_steps = args.rollout_steps
    device = torch.device(args.device)

    # Auto-enable amp only on CUDA
    use_amp = bool(args.amp and device.type == "cuda")
    amp_dtype = torch.bfloat16 if args.dtype == "bf16" else (torch.float16 if args.dtype == "fp16" else torch.float32)
    # fp16 needs GradScaler, bf16 does not
    scaler = torch.amp.GradScaler("cuda") if (use_amp and args.dtype == "fp16") else None

    policy = InternLMFinancePolicy(cfg, env.n_assets).to(device)
    # A5000 note: torch.compile needs Triton (Linux/WSL). On Windows it raises TritonMissing at first call.
    compile_enabled = False
    if args.compile and hasattr(torch, "compile"):
        # compile only trainable heads+encoders path; backbone is frozen - compile helps PPO loop
        try:
            policy = torch.compile(policy)
            compile_enabled = True
            print("torch.compile enabled (requires Triton; on Windows use WSL/Linux or omit --compile)")
        except Exception as e:
            print(f"torch.compile failed: {e}")

    optimizer = torch.optim.Adam([p for p in policy.parameters() if p.requires_grad], lr=ppo_cfg.learning_rate)
    obs, _ = env.reset(); steps, rewards = 0, []
    infos_rollout = []  # for detailed logging (net/gross/turnover)
    next_eval = args.eval_interval if args.eval_interval else 10**9
    next_ckpt = args.checkpoint_interval if args.checkpoint_interval else 10**9
    from pathlib import Path as _P; _P(args.output).mkdir(parents=True, exist_ok=True)
    print(f"A5000 config: device={device} dtype={args.dtype} amp={use_amp} compile={args.compile} rollout={ppo_cfg.rollout_steps} minibatch={ppo_cfg.minibatch_size} n_assets={env.n_assets} dates={len(env.dates)} | train_end={args.train_end or 'full'} eval_interval={args.eval_interval} ckpt_interval={args.checkpoint_interval}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)} | cap={torch.cuda.get_device_capability(0)} | mem={torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")
        # VRAM estimate: 1.8B bf16 ~3.6GB + LoRA + optimizer + batch
    while steps < args.timesteps:
        batch = {"obs": {k: [] for k in obs}, "actions": [], "log_probs": [], "rewards": [], "dones": [], "values": []}
        infos_rollout = []
        for _ in range(ppo_cfg.rollout_steps):
            with torch.no_grad():
                # autocast for inference too on A5000
                try:
                    if use_amp:
                        with torch.autocast(device_type="cuda", dtype=amp_dtype):
                            action, logp, value = policy.act(tensor_obs(obs, device))
                    else:
                        action, logp, value = policy.act(tensor_obs(obs, device))
                except Exception as e:
                    if compile_enabled and "Triton" in str(e):
                        print(f"torch.compile Triton missing -> falling back to eager (omit --compile on Windows): {e}")
                        policy = policy._orig_mod if hasattr(policy, "_orig_mod") else policy  # unwrap
                        compile_enabled = False
                        # retry eager
                        if use_amp:
                            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                                action, logp, value = policy.act(tensor_obs(obs, device))
                        else:
                            action, logp, value = policy.act(tensor_obs(obs, device))
                    else:
                        raise
            # A5000 bf16: action is bf16 -> numpy doesn't support bf16, cast to fp32
            next_obs, reward, terminated, truncated, info = env.step(action.squeeze(0).float().cpu().numpy())
            for k, v in obs.items(): batch["obs"][k].append(v)
            infos_rollout.append(info)
            # keep PPO tensors in fp32 for stable GAE/loss (bf16 -> fp32)
            batch["actions"].append(action.squeeze(0).float()); batch["log_probs"].append(logp.squeeze(0).float()); batch["rewards"].append(torch.tensor(reward, device=device)); batch["dones"].append(torch.tensor(float(terminated or truncated), device=device)); batch["values"].append(value.squeeze(0).float()); rewards.append(reward); obs = next_obs; steps += 1
            if terminated or truncated: obs, _ = env.reset()
            if steps >= args.timesteps: break
        with torch.no_grad():
            if use_amp:
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    _, bootstrap = policy(tensor_obs(obs, device))
            else:
                _, bootstrap = policy(tensor_obs(obs, device))
        values = torch.stack(batch["values"] + [bootstrap.squeeze(0).float()]); advantages, returns = compute_gae(torch.stack(batch["rewards"]), values, torch.stack(batch["dones"]), ppo_cfg.gamma, ppo_cfg.gae_lambda)
        batch["obs"] = {k:torch.as_tensor(np.stack(v), device=device) for k,v in batch["obs"].items()}; batch.update(actions=torch.stack(batch["actions"]), log_probs=torch.stack(batch["log_probs"]), advantages=advantages, returns=returns)
        ppo_update(policy, optimizer, batch, ppo_cfg, scaler=scaler, use_amp=use_amp)
        # detailed rollout stats (gross/net/turnover etc) — explains reward vs market
        m = _metrics(infos_rollout)
        print(f"steps={steps} mean_reward={np.mean(rewards[-len(batch['rewards']):]):.6f} mean_net={m.get('mean_net',0):.6f} max_dd={m.get('max_dd',0):.4f} mean_turn={m.get('mean_turnover',0):.4f} peak_equity={env.peak_equity:.2f} | rollout net/gross/turn")
        if device.type == "cuda":
            # optional VRAM log every rollout
            alloc = torch.cuda.memory_allocated(0)/1e9
            reserv = torch.cuda.memory_reserved(0)/1e9
            print(f"  vram alloc={alloc:.2f}GB reserved={reserv:.2f}GB")
        # periodic checkpoint + holdout eval (only if honest split train_end set, otherwise eval leaks)
        if steps >= next_ckpt:
            ckpt_dir = Path(args.output) / f"checkpoint_{steps}"
            policy.save_artifacts(ckpt_dir, serializable(cfg), serializable(ppo_cfg), env_cfg.__dict__, env.tickers)
            print(f"Checkpoint {steps} -> {ckpt_dir}")
            next_ckpt += args.checkpoint_interval
        if steps >= next_eval and args.train_end:
            try:
                from evaluate_policy import metrics_from_infos, run_episode, tensor_obs as _to
                # quick holdout eval using current policy (deterministic)
                eval_env = PortfolioEnv(args.data, window=args.window, start_date=args.eval_start, end_date=args.eval_end)
                eval_env.reward_config = env_cfg  # same costs
                import torch as _t
                eval_infos = run_episode(policy, eval_env, device, use_amp if device.type=="cuda" else False, amp_dtype, deterministic=True)
                eval_env.close()
                mm = metrics_from_infos(eval_infos)
                # baseline for same period
                bl_env = PortfolioEnv(args.data, window=args.window, start_date=args.eval_start, end_date=args.eval_end)
                bl_obs,_ = bl_env.reset(); bl_infos=[]
                while True:
                    _,_,term,trunc,info = bl_env.step(np.zeros(bl_env.n_assets, dtype=np.float32))
                    bl_infos.append(info)
                    if term or trunc: break
                bl_env.close(); bl_m = metrics_from_infos(bl_infos)
                alpha = (mm["final_equity"]-bl_m["final_equity"])/bl_m["final_equity"] if bl_m["final_equity"] else 0
                print(f"[Holdout {args.eval_start}->{args.eval_end}] learned {mm['final_equity']:.0f} (sharpe {mm['sharpe_proxy']:.2f} dd {mm['max_drawdown']:.2%}) vs equal_weight {bl_m['final_equity']:.0f} (sharpe {bl_m['sharpe_proxy']:.2f}) alpha {alpha:+.2%} @ {steps} steps")
            except Exception as e:
                print(f"[Holdout eval skipped] {e}")
            next_eval += args.eval_interval
    env.close(); policy.save_artifacts(args.output, serializable(cfg), serializable(ppo_cfg), env_cfg.__dict__, env.tickers)
    print(f"Saved to {args.output}: finance_lora/ + ppo_heads.pt + json")
    # final holdout print if honest split
    if args.train_end:
        print(f"Honest split: train_end={args.train_end} holdout={args.eval_start}->{args.eval_end} — run evaluate_policy.py for full report:")
        print(f"  python evaluate_policy.py --data {args.data} --artifacts {args.output} --start {args.eval_start} --end {args.eval_end} --device {device}")
