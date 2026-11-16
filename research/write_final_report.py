"""Write the final evidence report from frozen outcomes, without model selection."""
import argparse
import json
from pathlib import Path
import statistics

ROOT=Path(__file__).parent


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--results',required=True)
    p.add_argument('--analysis',required=True);p.add_argument('--output',required=True);args=p.parse_args()
    target=Path(args.output)
    if target.exists():raise FileExistsError(target)
    folder=Path(args.results).resolve();analysis=json.loads(Path(args.analysis).read_text());metrics=analysis['all_metrics']
    if json.loads((folder/'status.json').read_text())['status']!='completed':raise ValueError('Final block incomplete')
    protocol=json.loads((folder/'frozen_protocol.json').read_text())
    diagnostic=json.loads((ROOT/'campaign_runs/risk_strict_seed42_diagnostics/summary.json').read_text())['results']
    ablation_delta={key:diagnostic['lora_off'][key]-diagnostic['full'][key] for key in ('return_total','sharpe_proxy','max_drawdown')}
    neutral_delta=diagnostic['neutral_state']['sharpe_proxy']-diagnostic['full']['sharpe_proxy']
    def group(prefix):
        values=[metrics[f'{prefix}{s}'] for s in (42,123,456)]
        return {key:statistics.mean(v[key] for v in values) for key in
            ('return_total','return_annualized','sharpe_proxy','max_drawdown','mean_turnover')}
    cohorts=[('Primary direct-risk adapter, three-seed mean',group('candidate_seed')),
        ('Matched initialization, three-seed mean',group('initial_seed')),
        ('Independently trained no-LoRA, three-seed mean',group('trained_no_lora_seed')),
        ('Small Transformer, direct risk',group('numeric_risk_seed')),
        ('Small Transformer, matched-delay direct Sharpe',group('numeric_utility_lag1_seed')),
        ('Joint teacher reference, three-seed mean',group('teacher_reference_seed')),
        ('Numerical minimum variance',metrics['minimum_variance252_monthly']),
        ('Hierarchical risk parity',metrics['hrp252_monthly']),
        ('Daily equal weight',metrics['equal_weight_daily']),
        ('Fixed initial-risk target, monthly rebalancing',metrics['fixed_initial_risk252_monthly'])]
    comparisons=analysis['paired_comparisons'];dates=analysis['dates']
    lines=['# Finance adapter: completed local campaign','',
        '**Outcome: the adapter is technically functional, but this campaign did not meet the requested financial-quality goal.** On the frozen final period it underperformed both matched initialization and equal weight, with lower return and larger drawdown. The development improvement did not generalize. A finance-quality success or useful LLM advantage is not established.','',
        'A trained finance allocation specialist now runs on this PC against the fixed pristine InternVL3-2B. Training, saved-package reload and execution are verified. The table below reports the frozen post-publication test; it does not select a winner after seeing those results.','',
        f"Final return dates: **{dates['first']}–{dates['last']}**, {dates['count']} sessions. All market-dependent methods use one complete close of signal delay, fresh starting cash, identical fixed 49-stock data and 10 bps times traded-stock L1 costs. Sharpe uses risk-free rate zero. Returns are historical simulated results.",'',
        '| Method | Total net return | CAGR | Sharpe proxy | Max drawdown |',
        '|---|---:|---:|---:|---:|']
    for label,row in cohorts:
        lines.append(f"| {label} | {100*row['return_total']:.3f}% | {100*row['return_annualized']:.3f}% | {row['sharpe_proxy']:.4f} | {100*row['max_drawdown']:.3f}% |")
    lines += ['', 'Seed means describe three optimization runs on the same market history. They are not an ensemble portfolio or three independent markets. All individual models and the additional controls remain in the result JSON files.','',
        '| Primary seed | Selected training update | Total net return | Sharpe proxy | Max drawdown |',
        '|---|---:|---:|---:|---:|']
    choice=json.loads((ROOT/'METHOD_CHOICE.json').read_text())
    deployment_seed=choice['deployment_seed']
    selected_step_text='/'.join(str(choice['selected_steps'][str(s)]) for s in (42,123,456))
    development=json.loads((ROOT/'CAMPAIGN_RESULTS.json').read_text())['groups'][choice['selected_family']]
    for seed in (42,123,456):
        row=metrics[f'candidate_seed{seed}'];lines.append(f"| {seed} | {choice['selected_steps'][str(seed)]} | {100*row['return_total']:.3f}% | {row['sharpe_proxy']:.4f} | {100*row['max_drawdown']:.3f}% |")
    lines += ['',f"The deployment default is **seed {deployment_seed}**, selected by the same lowest pre-2023 risk-loss criterion across the three saved checkpoints before opening this final period. The paper's primary cohort remains all three seeds; seed 42 is a fixed technical/timing example. No final-period winner is promoted.",'', '## Differences and uncertainty','',
        '| Comparator | Primary minus comparator: mean ΔSharpe | Paired 95% interval, 20-session blocks |',
        '|---|---:|---:|']
    for name,row in comparisons.items():
        low,high=row['block_bootstrap']['20']['percentile_95_interval']
        lines.append(f"| {name} | {row['mean_seed_delta_sharpe']:+.4f} | [{low:+.4f}, {high:+.4f}] |")
    lines += ['', 'These are individual percentile intervals from 2,000 circular-block resamples of paired realized strategy returns. They do not retrain or rerun policies on synthetic prices. Five- and 60-session block sensitivities are also saved. Intervals containing zero do not prove equivalence or no effect; no family-wise comparison correction is claimed. One roughly nine-month historical holdout is limited evidence for future markets.','',
        'The separately specified adapter-only experiment is one seed with numeric modules frozen, compared with its own different initialization. It is a secondary attribution check, not a primary model eligible for post-test promotion.','',
        '## Objective, model and quality reference','',
        'The chosen method directly minimizes log portfolio variance relative to a uniform portfolio using a causal 252-return Ledoit–Wolf covariance. It uses the actual capped allocation, 30 sessions of five market features, covariance-conditioned numeric embeddings, all 28 original language-decoder layers, rank-16 attention LoRA (4,358,144 parameters), and a continuous allocation head. All base weights and vocabulary embeddings remain frozen. The full InternVL object is retained; no standalone smaller language model was substituted.','',
        'There are no optimizer-produced training labels or future-return rewards in this objective. Numerical constrained minimum variance supplies an external optimization-quality reference. It is **known-risk allocation, not PPO/GRPO, return forecasting or general financial QA**. The critic is unused. Allocations are fully invested and long-only, capped at 30% when rebalancing every 21 sessions; holdings may drift beyond the target cap between rebalances.','',
        'The objective is established [portfolio-risk mathematics](https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1540-6261.1952.tb01525.x), with the standard [Ledoit–Wolf estimator](https://www.ledoit.net/Well-conditioned2004.pdf). The numerical reference passed feasibility and scaled KKT checks on 32 pre-2023 holdout covariances (maximum residual 4.69e-6 against the declared 1e-4 tolerance). It is not an oracle for future realized risk.','']
    quality=analysis['allocation_quality']
    for seed in (42,123,456):
        q=quality[f'candidate_seed{seed}']
        lines.append(f"- Seed {seed}: mean volatility under the known decision covariance is **{100*(q['mean_volatility_to_numerical_optimizer']-1):.3f}% above the numerical optimizer**, across {q['rebalance_decisions']} actual rebalancing decisions.")
    lines += ['', 'This regret measures allocation quality for the estimated risk problem, separately from the realized return/Sharpe table. Each decision records both weight vectors and the covariance signal date. The model approximately learned the specified known-covariance risk problem, but that objective did not deliver the desired realized utility in this period. The numerical minimum-variance reference also lost money; following that reference is not sufficient for financial acceptance. This is an observed objective/generalization problem, not an excuse based on the GPU.','',
        '## Method choice and historical failures','',
        f'The economic direct-risk family was chosen before its legacy matched controls were inspected. A later numerical audit required strict-precision replication of the same fixed method/configuration, rather than a new financial-method search. Checkpoint selection uses only lowest mean log relative variance on 32 chronological pre-2023 holdout dates; strict selected updates are {selected_step_text}. All runs actually completed 512 updates at effective batch four. The exact choice record, artifacts, data, runtime, base contracts, installed environment and numerical flags were hashed before final outcome access. Seed 42 remains the fixed technical/timing example; the deployment default is chosen separately by the same pre-2023 risk criterion, never final returns.','',
        f"On reused zero-lag 2023–2024 development, the strict primary family averaged {development['mean_return_pct']:.4f}% return and Sharpe {development['mean_sharpe']:.4f}; mean drawdown change from initialization was {development['mean_drawdown_change_pp']:+.4f} percentage point. The numerical control scored 52.4099%, 2.2208 and 12.7055% drawdown. Legacy precision-permitted risk runs averaged 2.1511 Sharpe and their no-LoRA controls 2.1076; those are retained historical observations, not a relabelling of strict runs. Do not mix zero-delay development with the delayed final table.",'',
        'The original college runs had distinct problems: an initially reported smoke test took zero optimizer steps; BF16 PPO likelihoods were unstable; a functioning FP32 run suffered heavy turnover/costs; conservative/relative PPO stayed near its untrained diversified policy; future-return ranking failed chronological target validation. These findings do not show that a smaller GPU caused the financial failure.','',
        'The local inverse-volatility teacher learned its target but matched a tiny numeric model. Adding the same 252-session risk information as the stronger classical baseline produced a working joint teacher family. A larger frozen-head MLP LoRA pilot did not improve the result. Direct-Sharpe refinement reduced drawdown but did not improve the selected pilot’s return or Sharpe. These negative attempts remain recorded. The final direct-risk change is justified by the economic loss, not by VRAM or arbitrary novelty.','',
        f"Disabling LoRA after training and independently training a frozen-backbone control answer different questions. In the strict seed-42 post-hoc check, LoRA-off minus full is {ablation_delta['sharpe_proxy']:+.5f} Sharpe, {100*ablation_delta['return_total']:+.4f} return percentage points, and {100*ablation_delta['max_drawdown']:+.4f} drawdown points. Neutral-state minus full is {neutral_delta:+.5f} Sharpe. Report all these diagnostics alongside independently trained controls. Neither these checks nor a positive portfolio return establishes that the LLM is necessary.",'',
        '## Chronology, literature and scope','',
        'The exact local weight SHA matches the public hub LFS object last uploaded on **10 April 2025**; the [official model introduction](https://internvl.github.io/blog/2025-04-11-InternVL-3.0/) is dated 11 April. Therefore the original full-calendar-2025 plan was changed **before final-period strategy outcomes** to all available post-publication data, May 2025–30 January 2026. The protocol also adopts preceding-close signals on information-availability grounds, not by picking the favorable sensitivity value. Adapter training/optimization is pre-2023; 2023–2024 is reused development, not pretraining-disjoint evidence. Inaccessible college-PC test history and the complete pretraining corpus cannot be certified.','',
        '[FinATOM](https://arxiv.org/abs/2608.09880) is a close finance-LoRA analogue using teacher supervision then GRPO-style improvement; [direct Sharpe training](https://arxiv.org/abs/2005.13665) is also established. Our primary loss does not use the teacher-supervision/GRPO sequence; numeric covariance embeddings and a continuous head also differ. Attribute the prior work rather than treating changes as automatic proof of novelty. Published ETF/Nasdaq/clustered-Nifty headline Sharpes have different assets, dates, costs and inputs; they do not rank directly against this table. The local mathematical methods are adapted, matched controls, not exact reproductions of those papers’ original experiments.','',
        '[S-LoRA](https://arxiv.org/abs/2311.03285) and [LoraHub](https://arxiv.org/abs/2307.13269) show that multi-adapter backbones and cross-task adaptation have prior art. The particular four-domain combination alone is not a novelty argument. See LITERATURE_COMPARISON.md for the detailed ledger.','',
        'The stock universe is fixed after historical selection and is not reconstructed point-in-time Nifty membership. Corporate-action/download provenance is incomplete, although all 49 raw series cover the final date and there are no missing raw panel cells after November 2017. There is no order-book impact, spread/capacity model, tax or cash-allocation choice. Costs are 10 bp times full traded-stock L1, including initial cash entry, rather than 5 bp times half-L1 as in some papers. These are conditional historical research results, not a certified investable index or evidence of market alpha.','',
        '## Execution and portable evidence','']
    benchmark=json.loads((ROOT/'checks/runtime_benchmark_strict/result.json').read_text())
    router=json.loads((ROOT/'checks/router_smoke_deployment/result.json').read_text())
    precision=json.loads((ROOT/'checks/backend_precision/result.json').read_text())
    lines += [f"Real-package router check: **{router['status']}**; allocation difference from the simulator {router['allocation_max_difference']:.3g}, with current holdings preserved off schedule despite delayed signals.",'',
        f"Measured on this RTX 4060 Ti: GPU policy forward median **{benchmark['gpu_policy_forward']['median_ms']:.3f} ms**, complete router request **{benchmark['gpu_router_request']['median_ms']:.3f} ms**; CPU numeric Transformer **{benchmark['cpu_small_transformer_forward']['median_ms']:.3f} ms**, numerical optimizer **{benchmark['cpu_minimum_variance_solve']['median_ms']:.3f} ms**. These are 50 repetitions of one historical input, with CUDA synchronization and one math thread for CPU controls, excluding shared feature/covariance construction. Report the resource cost; shared-backbone use does not imply a standalone finance-compute advantage.",'',
        'The full-pretrained gate verified a real optimizer step, unchanged frozen-base tensor bytes, zero initial likelihood discrepancy and exact saved-adapter reload output. Main training peaks were around 5.0–5.1 GiB PyTorch allocated, excluding desktop/driver overhead. Inactive original components remain on CPU; active decoder tensors are FP32 with no rank/layer/context/asset reduction. Heavy compute pauses 22:50–07:00 IST, including real CPU-backbone execution; no fan or persistent power-plan settings were changed.','',
        f"The legacy inherited-permission sensitivity audit status is **{precision['status']}**: maximum score change {precision['maximum_score_difference']:.3g}, relative gradient L2 change {precision['relative_gradient_L2_difference']:.3g}, Sharpe change {precision['absolute_Sharpe_difference']:.3g}. That failed invariance check prompted replication; it was not waived. Final primary/matched-control training and neural inference use highest FP32 matrix-multiplication precision with **both TF32 permissions disabled**. The strict hardware gate passed. Auxiliary legacy references retain their recorded original training precision, with strict final inference. No bitwise cross-hardware training equality is claimed.",'',
        f"Frozen protocol: `{protocol['protocol_sha256']}`. Final evidence directory: `{folder.relative_to(ROOT).as_posix()}`. Complete candidate/initial/control packages, exact source/logs, daily returns, actual allocation vectors, uncertainty analyses and environment/base contracts are retained. Portable use requires the original pinned shared base; LoRA alone is insufficient without the saved numeric heads and execution schema.",'',
        'The new finance allocation component is complete for this experiment. General financial QA/news knowledge, live trading, real four-domain loading/switching/output parity, and comparative validation of the other specialists remain separate unfinished work. This campaign preserves the pristine base and does not certify those broader capabilities.','']
    target.parent.mkdir(parents=True,exist_ok=True);target.write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps({'report':str(target),'primary_mean':analysis['candidate_seed_mean'],
        'router_check':router['status'],'legacy_precision_audit':precision['status'],'strict_numerical_replication':True},indent=2))


if __name__=='__main__':main()
