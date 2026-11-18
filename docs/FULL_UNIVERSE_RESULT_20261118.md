# All49 run b completed: 18 November2026

96 RL iterations completed in7134seconds. Selected rl_1 by2024 validation; selected checkpoint exactly reproduced its validation equity path after reload. No training remains active.

Primary retrospective test:January2025-January2026. Net returns after declared fees/slippage.

| Capital INR | Selected RL | Untrained | Supervised | Holding | Equal rebalanced |
|---:|---:|---:|---:|---:|---:|
| 500 | 0.0000% | 0.0000% | 0.0000% | 0.0000% | 0.0000% |
| 5000 | 0.7220% | -0.3952% | 6.0691% | 0.0000% | 0.0000% |
| 50000 | 1.1516% | 0.7967% | 1.2885% | 3.6609% | 4.4012% |
| 100000 | 3.6663% | 3.5905% | 3.6711% | 6.3586% | 7.2038% |
| 200000 | 7.6030% | 5.4307% | 8.7744% | 8.0353% | 9.0824% |

At100000INR the RL gain over untrained is only0.0758percentage points; it trails supervised-only by0.0048points and holding by2.6923points. This does not meet the requested meaningful trained/RL advantage over both initialization and holding. At200000INR it improves over untrained but trails supervised and holding. The5000INR apparent win over holding must not be promoted: equal-weight holding purchased nothing under the small per-stock budgets, while supervised strongly outperformed RL.

The model produced14 distinct action vectors across14 primary-period decisions at100000INR. Counterfactual changes altered25 actions for small capital,20 for changed holdings,11 for reversed risk,12 for reversed returns; row reordering also changed8 actions. Responsiveness exists; row-order invariance remains unachieved. Neither proves good investment reasoning.

Execution exposure limitation: at the second primary-period decision, selected policy cash fraction56.43percent; holding50.24percent. Holding initially bought30 of49 assets because nominal equal-weight budgets could not buy a whole share of the others. All49 remain in the information/action universe; trading every stock is not affordable. The proportional shortage rounding can also leave cash. Therefore the current6.36percent holding result is not the previous fractional-share13.06percent result. This substantial execution consequence must be disclosed and audited before any improved allocation implementation is proposed; no test-driven silent benchmark change.

Later RL failed to improve the validation-selected first update; possible weak reward generalization and policy drift are hypotheses, not isolated causes. Do not claim hardware or2B parameter count proves an unavoidable performance ceiling. Original data and full-universe scope were retained, but publication controls and execution limitations remain. No candidate promoted to demo, no fresh run launched by this status check.
