# Candidate conclusion: Nifty native static adapter

Recommendation after explicit net-return baseline audit: Nifty native static seed42 checkpoint44 is a stronger candidate for a profit-improvement presentation than the current five-ETF risk-selector adapter. This is a retrospective recommendation, not a newly predeclared independent success. No deployment adapter or UI was changed by this audit.

| Period | Untrained net return | Trained net return | Equal-weight buy-and-hold net return |
|---|---:|---:|---:|
| 2023–2024 reused development | 44.110% | 45.618% | 45.387% |
| 2025 retrospective later-period audit | 13.274% | 15.074% | 14.864% |
| January2026 retrospective short interval | -1.915% | -1.614% | -1.588% |

The trained adapter beats both comparators in development and2025, but not buy-and-hold in January2026. Its buy-and-hold advantage is small: approximately0.23 and0.21 percentage points respectively, so do not claim a large or statistically established benefit. In2025 Sharpe: untrained1.103, trained1.243, buy-and-hold1.220; maximum drawdown8.135%,7.780%,7.750%. Thus trained return/Sharpe improve slightly over buy-and-hold, but its drawdown is slightly worse.

Native LoRA trains on2019–2021;2022 selects checkpoint44 using validation Sharpe;2023–2024 development was adaptively reused. All three existing training seeds increased development return versus their own untrained baseline; displayed later-period results are deployment candidate seed42 only. The later data were already accessed in prior project campaigns, and this candidate was revisited after seeing other failures: these new comparisons are retrospective robustness checks, not untouched confirmatory tests.

Fixed original49-stock Nifty universe and original data preprocessing are retained. Equal-weight buy-and-hold purchases all49 names once at the original first decision date, net of10bp fee using self-financing accounting, then holds fixed adjusted-price units. It has no later trades or terminal liquidation. Dates match each model path. Each period resets from100,000 cash; no stitched return is implied. Historical universe/preprocessing and point-in-time limitations remain. Saved constant-prompt action probabilities reproduce the adapter's financial policy without GPU generation; no model retraining was performed. A fresh actual-weight reload and portfolio replay plus stronger uncertainty/cost/universe checks are still needed before promoting this as the final deployed result. The present UI continues to show the old five-ETF adapter.

Sources: reports/static_native_seed_comparison_budget64.json; campaign_runs/nifty_static_native_seed42_64/development_0.json and development_44.json; original rl_candidate_static_native environment; data/nifty50_prices.csv; new nifty_static_buyhold.json and nifty_static_later_periods.json.
