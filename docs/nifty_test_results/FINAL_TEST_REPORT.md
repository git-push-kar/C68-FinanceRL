# Finance RL: verified candidate and training/validation/test results

## Recommendation

The Nifty native static strategy-preference adapter (seed42 checkpoint44) is the stronger candidate for demonstrating a training benefit. It improves test net return and Sharpe over the exact untrained policy, and slightly exceeds equal-weight buy-and-hold at the original10bp costs. Actual adapter weights were reloaded and verified. Do not claim it beats all simple trading strategies: rebalanced equal weight and inverse volatility remain stronger controls. No UI/deployment switch was made in this audit; the existing dashboard remains the old five-ETF experiment.

## Data partition for the final presentation

- Training:2019–2021, updates to LoRA parameters through financial RL.
- Validation:2022, checkpoint selection using net Sharpe. Selected steps42/123/456:44/40/44.
- Test:calendar2025 and January2026, shown separately. Also report a continuous1January2025–30January2026 test as supplementary material, carrying holdings through the year boundary.

2023–2024 development scores are EXCLUDED from the main results and figures. Their prior use for method design must remain disclosed in the methods/history. These test results are retrospective: the candidate was revisited after observing earlier experiments and the market data had been accessed elsewhere in the project. Changing terminology cannot turn them into independent untouched tests. Do not call validation or previously reused development history independent test evidence.

## Test2025:249 sessions,10bp fees

| Policy | Net return | Sharpe | Max drawdown |
|---|---:|---:|---:|
| Original untrained |13.274%|1.103|8.135%|
| Trained seed42 |15.074%|1.243|7.780%|
| Trained seed123 |15.074%|1.243|7.780%|
| Trained seed456 |15.074%|1.243|7.780%|
| Equal-weight buy-and-hold |14.864%|1.220|7.750%|
| Hold trained first target |14.741%|1.222|7.733%|
| Hold untrained first target |13.697%|1.136|8.120%|
| Equal-weight rebalanced |15.242%|1.244|7.787%|
| Inverse-volatility rebalanced |15.106%|1.252|7.754%|

The trained policy gains about1.80 percentage points versus untrained, and0.21 points versus equal-weight buy-and-hold. Its Sharpe gain over raw is about0.140. Its drawdown is slightly worse than buy-and-hold. All three trained seeds improve versus raw, but they share one market path and have near-identical strategies, so they are optimization replications, not independent market replications.

Conditional paired circular-block bootstrap,5000 resamples,20-session blocks: trained mean minus untrained return95% CI[0.590,3.264] percentage points; Sharpe CI[0.058,0.212]. Five/60-session sensitivity intervals also exclude zero for the raw comparison. These intervals do NOT adjust for historical method/candidate search and do not establish independent confirmatory significance. Versus equal-weight buy-and-hold, return CI[-0.880,1.277] points includes zero. Do not claim established superiority over holding.

## January2026 and continuous test

January2026 (21 sessions, independent cash reset): untrained-1.915%, trained-1.614%, buy-and-hold-1.588%. Training reduces the loss versus raw but does not beat holding. The month is too short for a strong standalone inference; bootstrap blocks comparable to or longer than this month are not credible independent confirmation.

Continuous1January2025–30January2026 (270 sessions, no year-end reset): untrained11.221%, trained13.283%, buy-and-hold13.059%, equal-weight rebalanced13.441%, inverse-volatility rebalanced13.343%. Report alongside the separate years; do not multiply reset-period returns and call them a continuous portfolio.

## Cost sensitivity

2025 test: at20bp, untrained12.563%, trained14.820%, buy-and-hold14.750%; at50bp, untrained10.458%, trained14.060%, buy-and-hold14.407%. The training gain over raw survives these modeled costs, but the small advantage over holding does not. These are fee sensitivity simulations, not a realistic market-impact model.

## What actually changed in the model

The same pinned full InternVL3-2B was loaded with FP32 active decoder and TF32 disabled; inactive original components remained on CPU. The six original packages (initial and selected for each seed) passed manifest verification. Maximum saved-versus-reloaded action-probability discrepancy was below1.5e-8. Disabling each trained adapter returned the same original untrained probabilities. This verifies real weight adaptation, rather than a manually substituted allocation.

For seed42, strategy probabilities changed approximately:
- Equal weight20.57% →32.88%.
- Inverse volatility31.39% →64.49%.
- Momentum30.48% →1.36%.
- Reversal17.56% →1.28%.

These probabilities are fixed strategy preferences; causal strategy target weights change with prices and the portfolio rebalances every five sessions. Training learned a more useful strategy mixture on these test histories. It did not learn daily news reasoning or prove state-dependent timing. The observed raw-relative gain includes lower turnover/costs, not exclusively superior gross stock selection.

## Verification and boundaries

Source stock-data SHA256 exactly matches the original training protocol. The original fixed49-stock universe and preprocessing remain; current-universe/survivorship and point-in-time limitations are not solved. All11 policies in each of nine period/cost combinations share exact return dates. Daily-return compounding and ending wealth reconcile; all hold baselines have zero turnover after the initial purchase. No training or original artifact modification occurred. Actual inference probabilities, full daily paths, bootstrap intervals, protocols and accounting checks accompany this report.

Main figures: presentation/nifty_test_2025.png, SVG and PDF; caption separate. Main presentation contains training/validation/test definitions and test results only. Historical development figures remain in prior archives for provenance.

## Defensible final statement

Financial RL changed the native2B model's strategy preferences and improved subsequent test return and Sharpe versus its original untrained policy across three training seeds. It slightly outperformed equal-weight buy-and-hold at the original fee setting over2025 and the continuous available test interval. That holding advantage is small, uncertain and cost-sensitive, and simple rebalanced strategies still match or outperform it. This supports an adaptation-benefit claim, not universal portfolio-management superiority.
