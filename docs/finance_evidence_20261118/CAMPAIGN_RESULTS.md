# Recorded campaign results

2026-11-07T16:48:05.555131+05:30

reused 2023–2024 development; lag1-trained utility controls use one-close delay, other rows retain zero lag; no final-period results

| Run | Step | Signal delay | Return | Sharpe proxy | Max drawdown | ΔSharpe vs initialization |
|---|---:|---:|---:|---:|---:|---:|
| covariance_lora_mlp_seed42 | 512 | 0 | 50.928% | 2.0783 | 14.222% | +0.4273 |
| covariance_lora_only_seed42 | 384 | 0 | 50.733% | 2.0953 | 13.167% | +0.6392 |
| covariance_no_lora_seed123 | 512 | 0 | 50.422% | 2.0898 | 12.292% | +0.4272 |
| covariance_no_lora_seed42 | 512 | 0 | 50.293% | 2.0757 | 13.329% | +0.4172 |
| covariance_no_lora_seed456 | 384 | 0 | 52.827% | 2.1136 | 13.859% | +0.4527 |
| covariance_risk_no_lora_seed123 | 384 | 0 | 51.667% | 2.1688 | 12.592% | +0.5062 |
| covariance_risk_no_lora_seed42 | 384 | 0 | 49.478% | 2.0902 | 13.291% | +0.4317 |
| covariance_risk_no_lora_seed456 | 384 | 0 | 48.875% | 2.0638 | 13.072% | +0.4029 |
| covariance_risk_seed123 | 512 | 0 | 53.899% | 2.2618 | 12.467% | +0.5992 |
| covariance_risk_seed42 | 384 | 0 | 51.317% | 2.1480 | 13.513% | +0.4895 |
| covariance_risk_seed456 | 512 | 0 | 48.055% | 2.0434 | 12.573% | +0.3825 |
| covariance_risk_strict_no_lora_seed123 | 384 | 0 | 51.710% | 2.1704 | 12.584% | +0.5079 |
| covariance_risk_strict_no_lora_seed42 | 384 | 0 | 48.802% | 2.0607 | 13.330% | +0.4023 |
| covariance_risk_strict_no_lora_seed456 | 384 | 0 | 48.849% | 2.0631 | 13.082% | +0.4021 |
| covariance_risk_strict_seed123 | 512 | 0 | 53.856% | 2.2599 | 12.477% | +0.5973 |
| covariance_risk_strict_seed42 | 384 | 0 | 50.211% | 2.1167 | 13.523% | +0.4582 |
| covariance_risk_strict_seed456 | 512 | 0 | 47.898% | 2.0366 | 12.585% | +0.3756 |
| covariance_teacher_seed123 | 512 | 0 | 51.013% | 2.1348 | 12.368% | +0.4723 |
| covariance_teacher_seed42 | 384 | 0 | 51.454% | 2.1249 | 13.248% | +0.4664 |
| covariance_teacher_seed456 | 384 | 0 | 52.139% | 2.1310 | 13.519% | +0.4701 |
| numeric_attention_seed123_lr0003 | 128 | 0 | 47.766% | 2.0549 | 13.447% | +0.3945 |
| numeric_attention_seed42_lr0003 | 512 | 0 | 47.798% | 1.9870 | 13.630% | +0.3258 |
| numeric_attention_seed42_lr001 | 256 | 0 | 48.649% | 2.0165 | 13.778% | +0.3553 |
| numeric_attention_seed456_lr0003 | 512 | 0 | 54.832% | 2.1930 | 13.593% | +0.5319 |
| numeric_covariance_seed123 | 256 | 0 | 50.820% | 2.1323 | 13.203% | +0.4715 |
| numeric_covariance_seed42 | 384 | 0 | 45.112% | 1.9382 | 13.527% | +0.2761 |
| numeric_covariance_seed456 | 512 | 0 | 48.375% | 2.0553 | 13.378% | +0.3951 |
| numeric_risk_seed123 | 384 | 0 | 49.053% | 2.0624 | 13.525% | +0.4020 |
| numeric_risk_seed42 | 512 | 0 | 52.429% | 2.1676 | 13.097% | +0.5064 |
| numeric_risk_seed456 | 512 | 0 | 49.495% | 2.0574 | 13.716% | +0.3963 |
| numeric_utility_lag1_seed123 | 32 | 1 | 46.830% | 2.0365 | 13.300% | -0.0594 |
| numeric_utility_lag1_seed42 | 64 | 1 | 43.915% | 1.8710 | 14.097% | -0.1389 |
| numeric_utility_lag1_seed456 | 128 | 1 | 51.431% | 2.1060 | 12.941% | -0.1106 |
| numeric_utility_seed123 | 96 | 0 | 44.958% | 1.9353 | 13.523% | -0.1197 |
| numeric_utility_seed42 | 64 | 0 | 43.366% | 1.8469 | 14.150% | -0.1401 |
| numeric_utility_seed456 | 128 | 0 | 50.687% | 2.0686 | 13.045% | -0.1243 |
| utility_seed42 | 32 | 0 | 50.122% | 2.0969 | 12.830% | -0.0280 |

Selection criteria and all checkpoints remain in each run folder.
Read LITERATURE_COMPARISON.md for published-method scope and classical controls.
This table alone does not establish statistical superiority or equivalence.
