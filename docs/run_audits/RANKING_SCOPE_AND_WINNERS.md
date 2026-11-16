# Historical extrema versus reportable test results

Inventory scope:195 campaign selection.json records with selected whole-development metrics;138 matched seed rows in the previous comparison index; plus frozen ETF additional yearly audit and new Nifty test audit. Not every intermediate training checkpoint, training fit, validation metric, teacher benchmark or retained-position yearly slice is ranked as a test result. Different periods/universes prevent a fair single global leaderboard. Historical maxima below are provenance, not eligible main results.

## Actual native RL adapters: later test results

- Highest test total return among the verified Nifty/ETF candidate comparisons: Nifty native static test2025, approximately15.074% versus13.274% raw. All three training seeds agree to rounding.
- Highest test Sharpe: ETF risk-selector native original reserved2026, trained three-seed mean1.951463 versus1.513051 raw. Individual seed1231.952345, but do not select the best seed from test scores.
- Lowest test maximum drawdown: ETF risk-selector native reserved2026, mean approximately2.4901% versus3.5094% raw. Individual seed1232.489045%; do not choose it retrospectively.
- Largest test net-return gain in these verified candidate comparisons: Nifty continuous1January2025–30January2026, approximately11.221% raw to13.283% trained, +2.062 percentage points. Calendar2025 alone improves1.800 points. Continuous and yearly reset simulations are separate, not independent replications.
- Largest test Sharpe gain in these verified candidate comparisons: ETF reserved2026,+0.438412 mean. Nifty2025 gains approximately0.140; continuous test approximately0.148.

## Historical high scores: excluded from main results

- Highest total return across195 archived selected-development records: numeric_attention_seed456_lr0003 checkpoint512,54.832%,2023–2024. Its own role is numeric-only teacher control,140801 parameters; not an InternVL RL adapter. Raw45.185%, gain9.647 percentage points. This is also the largest raw-relative return gain in that archived selection inventory.
- Covariance-risk strict seed123 checkpoint512:53.856% return and2.2599 Sharpe on2023–2024 reused development; selected on pre2023 teacher/variance fidelity. Different training/precision/timing methodology, not the selected native RL method; the covariance campaign failed later test. It must not be promoted on its development maximum.
- Additional annual slice with frozen ETF native deployment seed42: calendar2024 fresh-from-cash evaluation12.057% return,2.810 Sharpe,2.481% drawdown. It is reused development, not a final test score; it is not an independently trained2024 run.
- Highest total return among native strategy-PPO seed rows in the138-row comparison index: strategy_ppo_internvl_seed123_128 selected128,45.733% on2023–2024. Its whole-package PPO gain did not cleanly isolate incremental LoRA benefit; matched frozen-LoRA controls in the historical report remain necessary. Do not select seed123 because of the observed maximum.
- Largest archived Sharpe increase in195 selections: covariance_lora_only_seed42,+0.639243. This belongs to the older causal covariance-teacher campaign, not proof of a profitable native RL result.

Recommendation remains Nifty native static for demonstrating verified training benefit versus raw on later tests, with small/uncertain/cost-sensitive buy-and-hold advantage and stronger simple rebalanced baselines disclosed. No run is best across all metrics, market intervals and objectives.
