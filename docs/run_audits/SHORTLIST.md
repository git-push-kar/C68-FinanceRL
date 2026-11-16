# Historical runs with net-return improvement

Retrospective search of the saved comparison index. These are reused development comparisons, not new independent tests. Three seeds share the same market history. Numeric-policy controls are not evidence of InternVL adapter improvement. Return gain is percentage points, not relative percent.

| Report | Seeds with return gain | Untrained return | Trained mean return | Gain (pp) | Initial Sharpe | Trained mean Sharpe | Dates |
|---|---:|---:|---:|---:|---:|---:|---|
| static_native_seed_comparison_budget64.json | 3/3 | 44.110% | 45.617% | 1.507 | 1.581 | 1.698 | 2023-01-02 to 2024-12-31 |
| internvl_ppo_seed_comparison_budget128.json | 3/3 | 44.233% | 45.041% | 0.808 | 1.583 | 1.632 | 2023-01-02 to 2024-12-31 |
| multiasset_dynamic_native_comparison16.json | 3/3 | 24.938% | 25.453% | 0.515 | 0.884 | 0.989 | 2022-01-03 to 2024-12-31 |
| readout_only_ablation_comparison16.json | 3/3 | 24.938% | 25.519% | 0.580 | 0.884 | 0.955 | 2022-01-03 to 2024-12-31 |
| risk_selector_dynamic_comparison64.json | 3/3 | 25.468% | 25.992% | 0.524 | 1.006 | 1.526 | 2022-01-03 to 2024-12-31 |
| exposure_numeric_seed_comparison_budget1024.json | 3/3 | 15.873% | 18.927% | 3.054 | 0.565 | 0.562 | 2022-01-03 to 2024-12-31 |

Native Nifty static: direct financial RL on the original token strategy preferences; real InternVL LoRA. Native ETF dynamic: real state-conditioned InternVL LoRA. Readout-only: restricted adapter ablation, not full model attention adaptation. Dynamic risk selector and exposure numeric: CPU numerical-policy controls, not the requested final native adapter.

63 of 138 indexed seed rows have strictly positive net-return change. This index includes label controls, warmup continuations, numerical policies and small/mixed improvements, so the count is not a success rate for independent LLM experiments.

These candidates should not replace the final adapter merely because their development profits are higher. They require matched-date buy-and-hold/classical baselines and explicit later-period evaluation. The original final result is preserved.