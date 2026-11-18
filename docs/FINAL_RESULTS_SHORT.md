# Final Finance RL Result

**Selected adapter:** InternVL3-2B Nifty strategy-preference RL — seed42, checkpoint44.

- **Training:** 2019–2021
- **Validation:** 2022
- **Test:** 1 January 2025–30 January 2026, 270 trading sessions
- **Setup:** 49 stocks, five-session rebalancing, 0.10% fee per amount bought or sold; starting capital 100,000.

| Metric | Untrained | Trained | Buy-and-hold |
|---|---:|---:|---:|
| Net return | 11.22% | **13.28%** | 13.06% |
| Ending capital | 111,220.51 | **113,282.89** | 113,059.10 |
| Sharpe | 0.885 | **1.033** | 1.012 |
| Maximum drawdown | 8.13% | **7.78%** | 7.75% |

**Finding:** Training increased net return by **2.06 percentage points over untrained**. All three training seeds showed essentially the same improvement. Reloading the actual adapter reproduced its policy; disabling it restored the untrained policy.

**Limits:** The test is retrospective. The **0.22-point advantage over buy-and-hold is small and not statistically established**. Rebalanced equal weight returned **13.44%**, exceeding the trained policy. The dashboard now runs this Nifty adapter in an autonomous historical paper simulation. Conversation was removed after factual testing; the inspector shows actual execution records.
# Status correction — 17 November 2026

The figures below describe the **historical fixed-strategy-mixture adapter**. It does not satisfy the state-aware portfolio-management requirement and is not the newly concluded Finance result. The replacement is in progress; see [current training status](STATEFUL_RL_REVISION2_ACTIVE_20261117.md).

