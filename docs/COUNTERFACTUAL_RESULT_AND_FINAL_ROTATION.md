# Three-seed counterfactual results and final rotation experiment

All three InternVL3-2B seeds completed 1,536 financial updates. The fixed 2023–2024 validation criterion selected initialization for every seed. The January 2025–January 2026 retrospective results below use ₹100,000 and all 49 stocks.

| Model | Seed | Selected stage | Selected return | Untrained | Supervised |
|---|---:|---|---:|---:|---:|
| InternVL3-2B | 42 | untrained | 7.76% | 7.76% | 7.39% |
| InternVL3-2B | 123 | untrained | 7.76% | 7.76% | 6.35% |
| InternVL3-2B | 456 | untrained | 7.76% | 7.76% | 6.42% |
| numeric_control | 42 | untrained | 8.30% | 8.30% | 8.27% |
| numeric_control | 123 | rl | 6.61% | 6.90% | 7.78% |
| numeric_control | 456 | untrained | 14.29% | 14.29% | 5.69% |

Holding returned 12.56%, equal-weight rebalancing 11.11%, and the momentum teacher 5.85%. The selected model trailed holding by 4.80 percentage points. All three supervised policies beat the teacher on this period but trailed both holding and initialization. The selected-policy ablations for the native runs operate on initialization and do not establish a trained LoRA contribution.

The 2023–2024 trade-off already appeared during selection. The initialization was particularly strong in 2023. The teacher itself underperformed holding in the later test, and the financial updates did not produce a validation winner. These comparisons establish outcomes. They cannot isolate how much loss came from the teacher, the 20-session objective, or the information content of trailing prices. Earlier floor-only execution results are a different benchmark.

## Final experiment

Start each policy and holding with exactly the same costed, whole-share basket. Every 20 sessions InternVL scores all 49 stocks and can keep the portfolio or rotate at most 2% of equity between one funded pair. Other stocks keep their shares. An infeasible pair executes neither leg. Both annual validation equity paths reproduced the matched holding ledger exactly with rotation disabled.

Three fresh seeds each receive 768 financial updates directly from simulator excess log growth versus keeping the same inventory, with 16 sampled actions per state. There is no momentum teacher. The 2% cap and fixed no-trade threshold are design choices which limit turnover and potential upside. Small accounts may have few executable pairs. Existing positions can drift above the cap while new buys are constrained. Training ends in 2022. Select the best **RL** checkpoint per seed using 2023–2024 mean annual log growth, and compare each directly with initialization and holding regardless of validation outcome. Report every seed. The existing test period has informed multiple revisions. Any later favorable result still needs prospective confirmation.

Run location: C:/Users/Parth/Documents/PolyvalentFinance/runs/final_rotation49_20261118.
