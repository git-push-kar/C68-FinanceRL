# Finance RL: final research, deployment and failure audit

16 November 2026. This report supersedes earlier claims about which Finance artifact is final. It distinguishes measured results, implementation checks, interpretations and unresolved limitations. It does **not** establish a flawless process, a globally optimal policy, exhausted 2B-model potential or journal acceptance. The complete historical record remains in [CONTEXT_TRANSFER_FULL.md](CONTEXT_TRANSFER_FULL.md) and the recovery archives.

## 1. Selected artifact and one headline result

The selected research and paper-trading policy is the original **InternVL3-2B Nifty native static strategy-preference LoRA, seed42, checkpoint44**. Its weights have SHA256 `8b678cad4b5b8c100a717deccefd8825e72520c78e8aa823c18dfcdcc0dd0a35`. The exported adapter contains **6,809,504 parameters in 226 tensors**. Seeds123/456 selected checkpoints40/44 are replications, not alternatives picked for a favorable test score.

The headline interval is **1 January 2025–30 January 2026**, 270 daily return observations, continuously invested from 100,000 cash at the 31 December 2024 execution close. Separate 2025 and January2026 tests support the interpretation; they do not replace the continuous test or create independent market replications.

| Continuous retrospective test | Untrained | Selected trained | Equal-weight buy-and-hold |
|---|---:|---:|---:|
| Net return | 11.2205% | 13.2829% | 13.0591% |
| Ending capital | 111,220.51 | 113,282.89 | 113,059.10 |
| Sharpe, zero risk-free | 0.884590 | 1.033347 | 1.012044 |
| Maximum drawdown | 8.1348% | 7.7796% | 7.7499% |
| Total modeled fees | 699.26 | 240.41 | 99.90 |

Training improves return by **2.0624 percentage points over its original policy**. Its **0.2238-point advantage over holding is small and uncertain**. Rebalanced equal weight returns **13.4406%**, while inverse-volatility rebalancing returns **13.3429%**; both beat the selected trained return. The correct claim is an observed adaptation benefit over the model's own starting policy, not conventional-strategy superiority.

Currency in the active Nifty demo is INR. Simulated capital and prices are fractional adjusted-price exposures, not executable whole-share lots. Model size does not justify mislabeling the kind of decisions made or hiding stronger baselines.

## 2. Data, split and selection history

| Role | Dates | Use |
|---|---|---|
| Training | 2019–2021 | Update LoRA using financial reward |
| Validation | 2022 | Select checkpoint by net Sharpe; step0 eligible |
| Historical method development | 2023–2024 | Repeatedly examined during experimentation; excluded from main results |
| Retrospective testing | 2025 and January2026 | Evaluate frozen selected policies, with full-period and separate-year views |

The later data had been accessed in other project campaigns, and this candidate was revisited after the ETF result disappointed the profit-oriented criterion. These are **retrospective tests**, not untouched confirmatory evidence for the whole search. Excluding development scores from the main presentation does not erase their role in method design. It would be false to relabel those years as independent test observations.

The original fixed 49-name Nifty stock universe, vendor adjustments and preprocessing are retained. The source CSV's SHA256 matches the original training protocol. The demo's derived NPZ has its own checksum and contains 2023–January2026 for warmup and testing. The active engine reads only the required lagged windows and reveals account history only through the advanced session.

Current-universe/survivorship effects, retrospective vendor adjustments and historical missing-price forward filling remain limitations. This is not point-in-time constituent/fundamental data. A future point-in-time replication would be a new study, not an improvement that can be asserted without collecting and testing those data.

## 3. Exact model and financial-learning contract

The full pinned `OpenGVLab/InternVL3-2B` backbone uses revision `899155015275a9b7338c7f4677e19c784e0e5a21`. Rank16 LoRA updates attention and the original vocabulary projection; no replacement numerical actor head is used in this selected native method. The complete base remains present; the unused vision component stays on CPU during Finance text inference. Active language inference is FP32 with TF32 disabled. This is not layer removal or active-weight quantization.

The constant prompt describes four allocation strategies and requests the original vocabulary tokens A/B/C/D. Probabilities are `0.95 * softmax(logits / 2) + 0.05 / 4`. Both initial and trained policies use the same temperature and exploration. The raw baseline is the actual original model's probabilities, not a deliberately weakened or uniform initialization.

The reward objective is **negative training-path net Sharpe plus 0.01 KL regularization toward the original probabilities**. AdamW uses `1e-5` for attention LoRA and `3e-4` for vocabulary-head LoRA, with no weight decay. The budget is64 updates and validation every4 updates, with step0 eligible and earliest tie preference. Validation selects44/40/44 for seeds42/123/456.

This method propagates a financial path objective through the finite historical allocation/execution calculation. It is **direct financial reinforcement**, not the PPO used in earlier experiments. The local piecewise derivatives are not an unbiased population-policy-gradient estimator or proof of global optimization. The selected adapter does not use supervised stock-action labels, a news oracle or future test targets for training.

Optimizing Sharpe rather than ending wealth explains why another candidate could reduce risk while giving up profit. That is an objective mismatch with a profit-first user criterion, not a reason to call lower return a profit improvement.

### Causal allocation tools

For execution at close `t`, all features stop at close `t-1`:

- **A:** equal stock weights.
- **B:** half equal weight plus half normalized inverse trailing126-session daily volatility.
- **C:** half equal weight plus half standardized126-session momentum, excluding the latest21 sessions.
- **D:** half equal weight plus half standardized negative five-session returns, a reversal tilt.

Standardized tilts are clipped to[-2,2], exponentiated and normalized. The model's four probabilities linearly combine these stock targets. Targets change as lagged prices change; holdings drift between rebalances.

Seed42 preferences change approximately as follows:

| Strategy | Original | Trained |
|---|---:|---:|
| Equal weight | 20.574% | 32.880% |
| Inverse volatility | 31.388% | 64.487% |
| Momentum | 30.482% | 1.358% |
| Reversal | 17.557% | 1.275% |

The model learned a more useful **stable strategy mixture** on the measured histories. It does not read a fresh company event each day, and its probabilities do not adapt to state. Repeated live inference in the demo demonstrates deployment, not a new market-timing capability.

## 4. Execution, accounting and metrics

The account starts in cash. On every fifth session, targets are computed from lagged inputs and executed at the modeled close. A fee of10bp applies to the full purchase-plus-sale stock notional. The self-financing equation is `c = k * sum(abs((1-c)*target - old_stock_weights))`. Fees are funded from the portfolio, not free outside cash. Positions are marked to the next close; intermediate days preserve the actual drifted holdings.

Untrained and trained use identical dates, tools, scheduling, causal lag and costs. Buy-and-hold makes one post-fee equal purchase and holds fixed adjusted-price units thereafter. It is different from periodically rebalanced equal weight. Additional hold-the-first-trained-target and hold-the-first-untrained-target controls separate initial composition from subsequent rebalancing. No policy has a terminal liquidation charge. There is no leverage, shorting, modeled slippage, market impact, tax or real broker fill.

- Net return: ending wealth divided by starting capital minus1.
- Annualized return: geometric scaling with252 sessions/year.
- Sharpe: `sqrt(252) * mean(daily net return) / population standard deviation`, zero risk-free benchmark.
- Maximum drawdown: largest decline from the running peak, including initial cash value.
- Annualized volatility: daily population standard deviation times `sqrt(252)`.
- Mean turnover: daily average full-L1 stock turnover, including the initial purchase and zero-trade days.
- Total fees: sum of each session's fee fraction times that session's starting equity; fees are already in net return.

Cumulative per-stock market P&L includes earlier holdings and is before portfolio fees. It is not an individual transaction's realized tax gain, nor the amount bought or sold. The new engine verifies that market P&L minus fees equals the account's gain.

## 5. Test results, replications and uncertainty

Full policy tables for all three intervals and all10/20/50bp cases are generated directly from the saved metrics in [NUMERIC_APPENDIX.md](NUMERIC_APPENDIX.md). Full daily observations and additional controls are in `nifty_test_results/`. No test score in the main presentation is drawn from 2023–2024 development.

Calendar2025,249 sessions: raw13.274%, trained15.074%, equal holding14.864%; Sharpe1.103/1.243/1.220; drawdown8.135%/7.780%/7.750%. Rebalanced equal weight15.242% and inverse volatility15.106% remain stronger return controls. Hold trained first target14.741% and hold raw first target13.697% do not match the selected active portfolio.

January2026,21 sessions with independent cash reset: raw−1.915%, trained−1.614%, holding−1.588%. Training reduces loss versus raw but does not beat holding. The month is too short for a strong standalone inference. Bootstrap blocks as long as or longer than this sample cannot create credible independent monthly confirmation.

The continuous270-session path carries holdings across the year boundary. It is not obtained by multiplying independently reset annual returns. All three trained seeds give very similar positive changes versus raw; they show optimizer repeatability, not three independent markets.

Paired circular-block bootstrap uses common dates for policies,5000 resamples and5/20/60-session sensitivity. Calendar2025 trained mean minus raw return20-session95% interval is[0.590,3.264]percentage points; Sharpe interval[0.058,0.212]. The continuous return interval is[0.766,3.596]points. Holding-difference intervals include zero. These are conditional on the selected experiments and are **not adjusted for adaptive historical method search**. Do not claim independent confirmatory significance or established superiority over holding.

At20bp in2025, returns are raw12.563%, trained14.820%, holding14.750%. At50bp, raw10.458%, trained14.060%, holding14.407%. The raw-relative improvement survives these modeled costs; the small holding advantage does not. Fee sensitivity is not a nonlinear liquidity/impact model.

## 6. Why earlier results ranged from6% to50%+

They use different dates, lengths, universes, methods and evaluation roles. Nifty45.617% accumulated across two development years2023–2024. The same selected adapter returns15.074% in2025 and13.283% over the longer available test including January losses. The old ETF6.377% spans February–October2026 on five different ETFs. Their absolute returns cannot rank training quality on one common task.

The historical54.832% selected-development maximum belongs to a140801-parameter numerical teacher-control model, not the selected native InternVL RL adapter. Covariance-teacher scores and yearly slices retaining earlier positions must not be presented as independent native RL test records. Earlier large scores remain available as provenance, not headline evidence.

## 7. Approach chronology and supported failure interpretation

1. **Joint numerical encoders/heads and LoRA PPO:** established a working execution/training pipeline, with development returns around45.7%. Whole-package improvement did not isolate LoRA's incremental benefit. Matched frozen-LoRA controls left an incremental Sharpe difference around0.002, with an interval including zero. Large market profit alone was not evidence that the language adapter caused it.
2. **Covariance teacher/risk approaches:** produced attractive reused-development return/Sharpe. Some used zero lag, and a precision-invariance audit failed. Strict FP32/TF32-off replications were retained. The later lagged test failed: trained return approximately−0.839%, raw+9.455%. These teacher/risk branches were not promoted on their highest development score. They are not the selected native RL objective.
3. **Sector exposure and readout variants:** numerical/native policies tested restricted exposure, sampled actions, static mixtures and state-dependent readouts. Some sampled gains disappeared under deterministic deployment. Strong component-strategy correlation and ambiguous incremental model benefit limited conclusions. These observations do not establish that small models cannot learn financial policies.
4. **Native Nifty strategy-preference RL:** only LoRA changes original vocabulary strategy preferences. Development raw44.110% to trained45.617% was accompanied by worse drawdown11.932% to12.416%. That justified further evaluation, not an independent test claim. Later audits now show actual raw-relative benefits with explicit holding controls.
5. **Label, regime and warmup controls:** label permutations are controls, not distinct successes or weak-baseline selection opportunities. Regime/warmup continuations begin from specialized policies, so their initial score is not pristine untrained. Some incremental gains are tiny or worsen Sharpe; all are retained.
6. **Five-ETF direct/state/readout experiments:** train2010–2018, validate2019–2021, reused development2022–2024. Some native state-conditioned experiments had only16 updates; no convergence claim is made. The risk-mixture candidate improved reserved2026 mean Sharpe1.513→1.951 and drawdown3.509%→2.490%, while return fell6.532%→6.377%. Pure minimum variance Sharpe2.170 was stronger; the temporal improvement interval included zero. This is a risk-adaptation result, not proof of higher profit.
7. **Annual refits and contract audits:** yearly controls produced negative, zero and positive changes and failed their promotion screen. Separately reset years were not stitched into a purported investable path. Exact execution/funding, fresh reloads, disabled-adapter recovery and stronger baseline contracts addressed identifiable implementation/attribution weaknesses.
8. **Current Nifty selection:** revisited after a request for profit improvement and a holding comparison. It is explicitly retrospective, preserves the ETF result and creates no new pristine test. It better demonstrates raw-relative adaptation, not broad trading superiority.

Supported explanations for weak results include a narrow action menu, correlated underlying strategies/holdings, a constant policy incapable of market-state timing, a Sharpe rather than wealth objective, and trading-cost differences. These are observed structural/code/accounting facts. We have **not isolated2B size, hardware, dataset quality or one hyperparameter as the universal causal reason**. They are not proven excuses.

## 8. Explanation experiments and removal

The initial bounded/template engine did not satisfy the requirement for actual model responses and was removed. Five explanation SFT runs, synthetic data expansion, precision/decoder checks, prompting, model-authored correction and adapter-strength tests followed. Runs01–03 continued trading attention/vocabulary-head weights and generalized poorly despite low synthetic validation loss. Run04 restarted from pristine base with attention+MLP LoRA without vocabulary-head LoRA; run05 continued with retention/harder examples. Initialization, architecture and data changed together, so no isolated layer-level causal conclusion follows. Selected run05 checkpoint176 reached synthetic response validation CE0.004000088; this is not factual accuracy.

Earlier30-question BF16 development reviews gave run04 15pass/5partial/10fail and run05 15pass/4partial/11fail. FP32 improved some selected difficult probes but did not solve unfamiliar questions. Failed blank FP32 outputs caused by an input-device bug were classified as software failures and rerun after correction, rather than scored as poor model answers. All raw outputs remain archived.

The latest Nifty assessment saved and hashed30 cases before generation, using actual records at five portfolio dates. Questions covered buys/sells and false sale premises, event versus cumulative fees, strategy probabilities versus stock weights, drift, signed loss comparisons, P&L versus trade notional, individual-trade profit, signal dates, news/forecast/counterfactual limits and broker execution. The same concise prompt and facts were supplied to the selected explanation SFT and its disabled-adapter pristine base, both FP32. All60 real outputs and timing are retained. Review was by the assistant/developer, not independent or blinded.

The predeclared deployment gate required at least90% full passes and no material direction/accounting/fabricated-motive error on these common questions. Awkward wording alone did not fail an answer.

| Candidate | Pass | Partial | Fail |
|---|---:|---:|---:|
| Selected explanation SFT | 10 | 1 | 19 |
| Pristine base, same FP32 path | 8 | 1 | 21 |

Observed failures include a buy narrated as a sale, duplicated selected/cumulative fees, reversed negative P&L comparisons, portfolio profit attributed to a transaction, and orders falsely claimed sent to a broker. Some simple limitation answers were correct. The set is small and selected, so these counts are not universal model accuracy.

**Decision: remove Finance conversation.** No further narrative training or preset-answer replacement is claimed as a solution. The menu has no chatbot input or suggested-answer buttons; old explanation endpoints return410 without invoking generation. Its decision inspector displays explicit records and allocation calculations, not generated rationale or an internal thought transcript. Better future grounding/distillation might help, but no claim of impossibility is warranted. The user's2B allowance does not excuse material factual errors.

## 9. Autonomous historical paper-trading demonstration

Start/Pause, playback speed, Next day, Next rebalance and Reset advance a historical market. Each fifth session calls the full pinned base with the actual selected LoRA, computes the original four-row vocabulary projection and checks its output against the selected artifact fingerprint. It then computes causal targets and signed fee-funded purchases/sales. There is no stored-probability fallback for the trained path: model failure pauses it.

The constant policy gives essentially constant strategy preferences; changing prices and drift still change stock allocations/trades. Raw probabilities and holding are labeled **reference trajectories**, not fresh raw-model inference. The chart and records show only advanced history, not a prefilled final balance. The database does not contain a news KB, broker orders or a live market feed. Only a KB draft protocol was saved before the user stopped that proposal; no KB-conditioned performance is claimed.

The customer menu shows account value, cash, net gain, drawdown, fees, holdings and activity. Inspecting an event shows execution/cutoff dates, pre-trade wealth, amount traded, fee, strategy mix and individual buys/sells/targets. It explicitly distinguishes cumulative market P&L from a realized transaction gain. Filtering, exports and settings operate on revealed records.

Every one of the270 original saved daily equity values reproduces exactly in the new engine. A real API run made54 actual model calls, ending at113282.88926029207 with240.4142855807366 fees, zero daily-wealth difference from the original source outputs. A separate production headless-browser run used actual UI/API calls without route mocks, completed270 sessions, inspected a real event and rendered desktop/mobile layouts; fast playback took about55seconds on this PC. A distinct fixture suite tests transport/layout and is labeled as test fixtures, not new inference evidence.

## 10. Software verification and remaining risks

The current suite passes35 tests, with one upstream Starlette/TestClient deprecation warning. New checks cover golden daily path and raw/holding parity, fee funding/P&L reconciliation, future-price target invariance, capital scaling, scheduled-only model calls, failure before trade mutation and stale/double advance rejection. Existing residency and other-domain stub transport tests remain software tests; missing other-domain trained weights prevent certifying all four specialists' end-to-end quality.

Per-account locks and a global model lease serialize state/inference. Expected-session counters reject duplicate/stale advances. Playback generation tickets stop old loops resuming after pause/reset. Sessions are in-memory; server restart requires a new account. Missing weights or policy mismatch stops execution. Upstream base/weights must be installed; first model load is slower than warm calls.

These checks support reproducibility, not an assertion of no bugs or no lookahead in every historical experiment. Retrospective selection, fixed universe/preprocessing, limited regimes, simplified fills/costs, narrow actions and uncertain holding superiority remain material limitations. Stronger simple baselines directly contradict the claim that results cannot improve under the same circumstances. We therefore do not state that2B potential is exhausted.

## 11. Publication claims and external context

The defensible contribution is native small-model financial reward adaptation with explicit original-policy attribution, fair comparator accounting, uncertainty, unsuccessful variants and an actual deployable historical-paper workflow. Combining several adapters alone does not establish research novelty. No matched external benchmark reproduction or independent assessor was supplied.

- [Moody and Saffell, Learning to Trade via Direct Reinforcement](https://people.idsia.ch/~juergen/rnnaissance2003talks/MoodySaffellTNN01.pdf) provides the established financial direct-reinforcement context. Our finite-path method is not a new proof of unbiased/global optimization.
- [LoRA](https://arxiv.org/abs/2106.09685) supplies established parameter-efficient adaptation; rank16 is not an algorithmic novelty claim.
- [InternVL3](https://arxiv.org/abs/2504.10479) is the backbone reference, not evidence of this portfolio's quality.
- [FinRL](https://arxiv.org/abs/2011.09607) provides established finance-RL environment/evaluation context; the custom controller is not advertised as a reproduction of its agents.
- [FinATOM](https://arxiv.org/abs/2608.09880), *Financial Numerical Prediction and Allocation as Token Generation*, reports five-ETF2023–2025 pooled net Sharpe1.394→1.494 at5bp. Its dates, inputs, initialization and objective differ. Neither our Nifty return nor ETF Sharpe establishes a win over it. The methodological connection is constrained LM-token financial decisions. These primary references were reopened during this audit.

## 12. Final presentation shortlist

1. One artifact, exact train/validation/test dates, and the full continuous retrospective test as headline.
2. Trained/raw/holding comparison, with simple rebalanced controls, all seeds, uncertainty and cost sensitivity in supporting tables.
3. Actual model-backed autonomous paper playback and its recorded trades; explain the model/tool/execution boundary.
4. Explain that Finance conversation was removed after factual testing. Show trustworthy records without calling them model reasoning.
5. Conclude raw-relative adaptation benefit on measured history, not state-of-the-art returns, established holding superiority, news forecasting, a flawless process or exhausted model potential.

Recovery status, commit hashes, checkpoint-release verification and portable setup are maintained separately so an unfinished backup is never described as complete. See [START_HERE.md](START_HERE.md), the main Git history and `archives/` receipts. Historical reports may call earlier artifacts final; later dated decisions supersede those descriptions without erasing provenance.

## Interface addendum — 17 November 2026

Finance was integrated natively into the original Polyvalent Chat/Anomaly interface, preserving the other workspace markup and shared theme. The final production-browser run made 54 real adapter calls through 270 sessions and matched every original daily balance exactly (maximum difference 0.0 INR). UI integration, provenance and controls/theme checks are documented in [FINANCE_UI_INTEGRATION_20261117.md](FINANCE_UI_INTEGRATION_20261117.md). No training, weights, financial method or headline result changed; Finance generated conversation remains disabled.
