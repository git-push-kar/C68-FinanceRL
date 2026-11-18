# Finance RL experimental record and integration decision

18 November 2026 · InternVL3-2B · Indian equity portfolio simulation

At ₹100,000, all three final state-aware policies earned a positive net return and modestly improved on the same model with its adapters disabled. Holding the common opening basket earned more than every trained policy. We recommend publishing the experiment history and keeping the existing strategy-mixture demo under its original description. The final rotation adapters are research artifacts. None qualified to replace the demo.

## How to read the record

Return means cumulative change in simulated equity after the costs charged by that study. A percentage-point gap subtracts two returns. Development was inspected repeatedly. Validation chose checkpoints. Later revisions reused the 2025–January 2026 retrospective window, so the final window cannot certify generalization to unseen markets. Independent training seeds share market observations. They measure training variability on one history.

## Experiment chronology and decision trail

| Stage | Question and intervention | Evidence and decision |
|---|---|---|
| Training pipeline and joint PPO | Could financial rewards move parameters and alter trading? Early audits of optimizer updates, likelihood accounting and precision preceded comparisons of numerical heads plus LoRA. | Roughly 45.7% development return looked attractive. Matched frozen-LoRA controls left about 0.002 incremental Sharpe with uncertainty spanning zero. Attribution required a simpler action path and explicit raw controls. |
| Covariance, utility and risk teachers | Tested imitation, trainable subsets, no-LoRA, stricter arithmetic and delayed signals. | Several reused-development paths exceeded 50%. The 54.832% maximum belonged to a numerical teacher control. Precision sensitivity, comparable no-LoRA scores and a lagged test (trained −0.839%, raw +9.455%) weakened the case for promotion. |
| Sector exposure and native readouts | Compared deterministic and sampled actions, static strategy mixtures and conditional readouts. Added label permutations, warm starts and annual refits. | Some sampled gains disappeared on deterministic replay. Annual refits gave mixed signs. Specialized initialization and separate yearly resets preclude combining their best numbers into a continuous record. |
| Five-ETF risk adaptation | Tested direct policies and a risk-strategy mixture on a distinct five-ETF dataset. Training 2010–2018, validation 2019–2021, reused development 2022–2024. | Reserved 2026 mean Sharpe rose from 1.513 to 1.951 and drawdown fell from 3.509% to 2.490%. Return fell from 6.532% to 6.377%. Minimum variance reached 2.170 Sharpe. This motivated a separate profit and holding comparison. |
| Nifty strategy-preference adapter | Trained native strategy preferences in 2019–2021 and validated in 2022. The allocation tools consumed market data and the simulator tracked wealth. The model itself lacked stock and account observations. | Seed 42, checkpoint 44 returned 13.2829% against raw 11.2205%, holding 13.0591% and equal rebalancing about 13.44%. The mixture repeated across 54 decisions while downstream targets changed. The model knew the playbook before it knew the portfolio. This is a useful retrospective baseline for learning a strategy preference. It cannot establish account-aware stock selection. |
| Explanations and demo | Tested template output, then five explanation SFT runs. Runs 01–03 continued trading weights. Run 04 restarted on pristine base with a changed LoRA specification. Run 05 added harder examples and retention. | Low synthetic loss did not ensure factual explanations. The 30-question development reviews of runs 04 and 05 each yielded 15 passes. Finance conversation was disabled. The historical demo reproduces 270 saved equity points and 54 actual adapter calls. Show its holdings and execution records as records. |
| Stateful five-stock run 1 | Added account and stock state and direct stock actions, but narrowed the universe to five equities. | The first policy collapsed and was stopped. These experiments diagnose the decision interface on five names. The original full-universe requirement was missed. |
| Stateful run 2 | Revised the training and action path after collapse, then tested the validation-selected candidate. | +22.372% validation became −8.176% on the continuous January 2025–13 November 2026 extension, versus +1.893% raw and −1.795% holding. Seven sign perturbations changed greedy decisions only once. Selection on one good year gave little protection. |
| Stateful run 3 | Tested a further conditional-action revision and explicit perturbations. | On the extended test the selected policy returned −3.838%, raw −0.113%, holding −1.795%. All five targets were 20%. Six validation probes left greedy actions unchanged. Providing holdings and prices alone did not make the policy use them. |
| Stateful run 4 and fallback | Required conditional imitation before financial RL. The gate passed at 2,048 examples with 63.54% balanced accuracy versus raw 14.58%. A shutdown required a weights-only restart at step 64 with fresh Adam. | Step 64 reached 19.8887% on 2024 validation, versus 14.2273% holding. Work stopped to restore the 49-stock scope. The prepared run-5 input-bridge fallback was conditional on gate failure. It has no five-stock financial result here. |
| Full-universe rebuild, attempts a and b | Restored all 49 original stocks, numerical input bridge, native per-stock ratings, executable prices and whole-share costs. Attempt a stopped after its first update to correct target sizing while corporate-action receivables were outstanding. Pending value counts toward equity and cannot fund purchases. Run b completed 96 RL iterations. | At ₹100k validation chose RL update 1. Retrospective returns were RL 3.6663%, raw 3.5905%, SFT 3.6711% and holding 6.3586%. State perturbations changed actions, but row reordering also changed eight. Early floor-only sizing left around half the account in cash for holding at its second decision. This justified revisiting residual-cash allocation and canonical ordering. |
| Three-seed counterfactual credit | Improved whole-share execution, canonicalized stock inputs, and trained relative stock ratings using simulator counterfactuals. Each native seed received 1,536 financial updates. A separate numerical control tested dependence on the backbone. | Validation selected initialization for all three native seeds. On the primary period initialization earned 7.76%, SFT earned 7.39/6.35/6.42%, holding 12.56%, equal rebalancing 11.11% and the momentum teacher 5.85%. A secondary validation-best **RL-only** replay earned 7.00/6.55/5.89%. Excluding initialization from selection would not have rescued the result. |
| Three-seed sparse rotation | Removed momentum imitation and anchored decisions to an identical opening portfolio. Allowed one funded pair rotation or a hold every 20 sessions. This isolated the incremental value of active trades and limited turnover. | See all checkpoints, matched comparators, costs, capital sensitivity and uncertainty below. The final policies narrowed the holding deficit relative to the preceding design. Opening exposures and action spaces also changed, so the cross-study difference cannot be assigned to training alone. |

The five-ETF experiment and the later five-stock restriction are separate studies. The original panel contains 49 stocks through 30 January 2026. Later 2026 prices in the five-stock study belong to an extension. There is no hardware justification for the reduced stock universe in the final method. A GPU preflight subsequently exercised the complete 49-stock input and action path.

## Final rotation experiment

Training sampled July 2018 through 2022. The policy receives 25 causal features per stock through a learned numerical input bridge and scores all 49 names through the native model. Each seed took 768 financial updates with 16 sampled actions per state, using paired 20-session log growth relative to keeping the same inventory. A new purchase is capped at 2% of equity. Positions may drift. Selection used the highest mean annual validation log growth among RL checkpoints, with earliest ties. The evaluator reloaded each selected package and checked both annual validation returns before test replay. Raw disables LoRA and bypasses the numerical bridge, so the difference measures the joint adaptation.

**Primary test, ₹100,000.** The continuous window runs from January 2025 through 30 January 2026, covering 270 sessions. The opening share trades match between each selected policy, raw and holding. Raw and holding equity paths each agree across seeds. Rotations count decision dates with executed trades after the opening basket.

| Seed | Checkpoint | 2023 validation | 2024 validation | Test net return | Δ raw, pp | Δ hold, pp | Sharpe | Max drawdown | Fees, ₹ | Rotations |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | rl_672 | 29.67% | 15.68% | 10.426% | +0.052 | -2.138 | 0.782 | 11.31% | 138.34 | 13/13 |
| 123 | rl_480 | 29.65% | 15.68% | 10.535% | +0.161 | -2.030 | 0.788 | 11.28% | 138.59 | 13/13 |
| 456 | rl_480 | 29.75% | 15.74% | 10.513% | +0.138 | -2.052 | 0.789 | 11.28% | 140.21 | 13/13 |

**Starting allowance.** Each row uses identical initial cash for every policy. The execution ledger includes whole shares, fees, slippage and unspent cash.

| Capital, ₹ | Hold | Equal rebalanced | Momentum | Raw | RL 42 | RL 123 | RL 456 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 500 | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% |
| 5000 | 4.847% | 2.723% | -1.775% | 4.847% | 4.847% | 4.847% | 4.847% |
| 50000 | 15.456% | 13.267% | 1.372% | 12.094% | 11.708% | 12.366% | 12.366% |
| 100000 | 12.565% | 11.106% | 2.594% | 10.374% | 10.426% | 10.535% | 10.513% |
| 200000 | 13.592% | 12.281% | 3.342% | 11.362% | 11.211% | 11.250% | 11.103% |

At the primary capital each trained policy traded at all 13 opportunities after the common opening. Raw lagged holding by about 2.19 percentage points. The adapters recovered only 0.05–0.16 points of that gap. At ₹200,000 every trained seed trailed raw. The sign of the primary raw-relative result therefore depends on the account size.

At ₹500 the identical zero returns illustrate an affordability limit. Different final returns across larger accounts establish capital-sensitive simulated outcomes. A claim of learned capital-dependent strategy would require separating changes in model preferences from deterministic changes in executable shares and pair eligibility.

**Paired uncertainty.** Descriptive 95% circular block-bootstrap intervals below use 20-session blocks and 100 × cumulative log-growth difference. All intervals include zero. They omit model-search uncertainty, and three seeds on one market path do not provide three independent market tests.

| Seed | RL minus raw | RL minus holding |
|---:|---:|---:|
| 42 | -0.031 to +0.155 | -5.449 to +1.907 |
| 123 | -0.335 to +0.697 | -5.445 to +2.093 |
| 456 | -0.503 to +0.935 | -5.592 to +2.318 |

## Why the holding values differ

The historical Nifty demo used fractional allocation and returned 13.0591%. The first restored whole-share study floored per-stock budgets, initially buying 30 of 49 at ₹100,000. Its holding return was 6.3586%. The later counterfactual and rotation studies used the revised whole-share opening allocation and report 12.56% over the same nominal January 2025–January 2026 primary window. Five-stock studies also used a different basket and, in some comparisons, a longer window. Always compare a policy against holding from its own execution ledger at the same capital. A gap between historical baseline values is not a training improvement.

## Publication and integration decision

Report two findings separately. A learned static strategy preference improved on initialization in the historical demo. In the full-information experiment, all three selected rotation policies modestly improved on initialization, but the passive basket and equal rebalancing earned more at the primary capital. The repeated primary window, simplified costs, fixed universe, limited regimes and conditional execution constrain generalization. The current evidence cannot isolate model size, data quality or reward horizon as the cause of the deficit.

Keep the existing demo for the reproducible strategy-preference example. Archive all three final adapters for research replay. The final policy emits scores and orders, and there is no validated natural-language explanation from it. A further overnight sweep on the same dates would add another adaptively selected comparison. The next informative experiment would lock the policy, benchmark and capital before collecting a fresh period. Use the present timeline and full matched results for the paper.

## Evidence and reproduction map

- [Historical research report](JOURNAL_RESEARCH_REPORT.md), [full working chronology](CONTEXT_TRANSFER_FULL.md) and [numeric appendix](NUMERIC_APPENDIX.md) document historical choices, splits, controls and deployment audits.
- [Directory-by-directory campaign inventory](finance_evidence_20261118/campaign_directory_index.json) lists every archived campaign directory, available root records and extractable selected metrics. A directory may be a pilot, restart or control. Missing fields indicate missing or differently structured records. [Original development ledger](finance_evidence_20261118/CAMPAIGN_RESULTS.md) gives individual historical scores and signal delays.
- [First stateful study](STATEFUL_RL_ACTIVE_RUN_20261117.md), [second revision](STATEFUL_RL_REVISION2_ACTIVE_20261117.md), [third revision](STATEFUL_RL_REVISION3_RESEARCH_20261117.md), [fourth revision](STATEFUL_RL_REVISION4_ACTIVE_20261117.md) and [fallback protocol](STATEFUL_NUMERIC_BRIDGE_CONTINUATION_20261117.md) preserve the sequence.
- [Full-universe rebuild](FULL_UNIVERSE_CONTINUATION_20261117.md), [whole-share results](FULL_UNIVERSE_RESULT_20261118.md), [counterfactual protocol](COUNTERFACTUAL_FULL_CAMPAIGN_20261118.md), [native and numerical results](COUNTERFACTUAL_RESULT_AND_FINAL_ROTATION.md) and [secondary RL-only replay](finance_evidence_20261118/counterfactual_rl_only.json) document the successive changes.
- [Complete final metrics, candidates, protocol and bootstrap records](finance_evidence_20261118/rotation_records.json) and the [actual training and evaluation implementation](../research/final_rotation.py) support the tables. The portable archive contains per-day equity, executed share ledgers, base weights and checkpoint packages under `PolyvalentFinance/runs/final_rotation49_20261118` and `PolyvalentFinance/local_models`. Earlier reports retain the status at their original dates. This record supersedes their pending-run and promotion language.

For independent replay, use the [checkpoint loading instructions](ROTATION_REPLAY_20261118.md) and the [primary equity and execution ledgers](finance_evidence_20261118/).
