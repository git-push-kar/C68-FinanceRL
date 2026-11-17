# Finance RL: design review and third state-aware experiment

17 November 2026. Status at creation: **launched; no performance success claimed**.
This report supersedes the active-worker information in revision 2. Read the
actual worker status before taking any process action. The model and PC remain
InternVL3-2B and RTX 4060 Ti 8 GB. The user authorized a reasoned 12–18-hour
continuation, with some flexibility, not an indefinite search.

## The actual problem and what has been learned

The historical 13.28% static-mixture adapter was genuinely evaluated, but never
received the changing stock/account observations. It cannot demonstrate learned
portfolio-aware trading. Calling repeated model calls adaptive decisions was
incorrect. The user's objection was substantive, not cosmetic. A static
allocation model can be useful under a different claim, but is not this task.

State-aware run 1 supplied those inputs but collapsed to concentrated allocations.
It was stopped on validation failure. Run 2 added concentration limits, longer
holding/reward intervals, a corrected token-query endpoint, row-order shuffling
and conservative optimization. It selected step 1792 on 2024 validation.

At 100,000 INR, run 2 validation return was 22.372%, versus untrained 6.269%.
On the continuous January 2025–13 November 2026 retrospective test, however:

| Policy | Net return | Sharpe | Maximum drawdown |
|---|---:|---:|---:|
| Selected run 2 trained | −8.176% | −0.218 | 23.484% |
| Matched untrained | +1.893% | 0.355 | 4.311% |
| Equal-weight initial purchase then holding | −1.795% | −0.012 | 18.232% |

These results do **not** meet the user's success criterion. The large validation
gain did not generalize. Low exposure can also make the untrained model's small
positive return look attractive; exposure and risk must accompany returns.
The matched January 2025–January 2026 slice gave trained 2.867%, raw 3.097%,
holding 7.859%; it is not a rescue of the selected run.

Actual-model counterfactual checks on seven validation observations reversed
trailing-return signs while leaving stock identity, prices and account facts
fixed. Six of seven produced exactly the same discrete decisions; probability
changes were small but nonzero. This is evidence of weak decision responsiveness
in those probes, not proof that inputs were ignored everywhere. Changing all
signals creates synthetic states, so this diagnostic cannot establish a trading
edge. Artifact copies: `stateful_rl_run2_result/`.

Plausible failure mechanisms are a weak initial numerical policy, easy stock or
position shortcuts, optimizing isolated randomly generated account snapshots,
limited independent market histories, and validation regime overfitting. The
experiments do not isolate their individual causal contributions. There is no
evidence that the GPU itself caused bad investment decisions, nor proof that
2B parameters make the task impossible. More optimizer steps alone were not a
justified response to these failures.

## Research review and design choice

This is a targeted review of relevant primary work, not a claim to have read
countless papers or discovered the globally best method.

1. [FinRL](https://arxiv.org/abs/2011.09607): practical financial RL needs a
   market environment, trading constraints and reproducible backtests. This
   supports treating execution/account state as essential infrastructure.
2. [Jiang, Xu and Liang](https://arxiv.org/abs/1706.10059): sequential portfolio
   allocation and prior portfolio state are central. Their crypto returns and
   frequency are not comparable to this five-stock daily experiment.
3. [PPO](https://arxiv.org/abs/1707.06347): bounded policy updates provide the
   optimization basis. We do not claim to implement its full actor-critic form.
4. [DeepSeekMath](https://arxiv.org/abs/2402.03300): group-relative policy
   optimization provides a critic-free reference. Our groups contain financial
   trajectories, not reasoning answers; this is an adaptation, not reproduction
   of their results or an assertion that math gains imply finance gains.
5. [FinATOM](https://arxiv.org/abs/2608.09880): directly relevant token-based
   financial allocation uses a causal supervised anchor followed by policy
   optimization. This motivates a useful initialization before reward learning.
   We use a simpler volatility-based curriculum, integer shares, explicit account
   observations and multi-decision account trajectories. Their ETF data,
   information set, costs and model are different; their metrics cannot be
   used as a matched leaderboard. It is a preprint, not proof of our design.
6. [Generalizable trade execution, IJCAI 2023](https://ijcai.org/proceedings/2023/553):
   limited market contexts can produce overfitting even in execution RL. This
   motivates simple causal feature summaries and conservative interpretation.
   Execution optimization is a different task from choosing investments.
7. [FinMem](https://arxiv.org/abs/2311.13743): memory and information-processing
   architecture can matter, but adding news/retrieval infrastructure now would
   change the information set without fixing the demonstrated policy-learning
   failure. No KB or invented financial narrative is added to this run.
8. [Can DRL beat 1/N?](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5130133):
   broad evaluation finds that sophisticated RL does not systematically beat
   simple diversification and that costs matter. Buy-and-hold is a substantive
   competitor, not "nothing" that a valid learner must automatically beat.

No cosmetic algorithm change creates research novelty. The defensible prospective
contribution is a measured adaptation and audit of a small native language-model
adapter for stateful, budget-constrained trading. Its strength depends on the
actual ablations and results. It cannot be declared journal-worthy in advance.

## Chosen experiment and reasons

### 1. Causal supervised allocation curriculum

Three explicitly coded candidate teachers were evaluated annually on 2015–2023:
inverse volatility, volatility-weighted trend tilt, and absolute-trend top-three
selection. The selection score is mean annual net log growth minus 0.1 times
annual drawdown. Scores were 0.120462, 0.109080 and 0.046413 respectively.
**Inverse volatility was selected using training years only.** The 2024 results
are also recorded but did not select the teacher. Its quantized validation
return was 19.123%, a diagnostic reference, not a trained-model result.

This teacher assigns more weight to lower recent volatility, excludes shares
unaffordable from total equity, quantizes to the policy action vocabulary, and
uses a 2.5-percentage-point existing-weight deadband for KEEP. It is deliberately
simple and does not predict future returns. The observed best teacher is not
promised to beat holding later. The three-candidate choice is itself recorded
as a method-selection decision.

768 valid training-date account states span 500/5,000/50,000/100,000/200,000 INR,
cash-only and random existing inventories. Two supervised passes use only causal
teacher labels, 5% label smoothing, shuffled rows and IDs. The inventory contains
179 distinct action vectors; the largest single-action class is 46.56%.
This reduces the burden of discovering portfolio allocation from the language
model's arbitrary initial letter preferences. No future outcomes create labels.

The selected supervised checkpoint is retained as a separate comparator. Its
selection uses 2024 net growth, not test returns. The teacher is NEVER an
inference fallback or a hidden blend with the trained model.

### 2. Own-account episodic RL

Each group samples four 60-session trajectories from the same training starting
date and capital. Each trajectory takes three model decisions, spaced 20 sessions
apart. Trades alter cash and holdings, and those updated quantities enter the
next observation. Two groups are collected per update, at most 128 updates.
Starting-date buckets balance up/down trailing trends and high/low historical
volatility; bucket definitions use only data known at the starting date.

Reward is terminal net log wealth growth minus 0.1 maximum drawdown, after fees
and slippage. Group-centered, variance-scaled rewards train a clipped joint
five-action policy gradient. The standard-deviation floor is 0.01; ratio clipping
is 0.8–1.2; reference KL coefficient 0.01; entropy coefficient 0.003. The selected
supervised policy supplies a frozen reference. Two optimization passes are made
per collection, with a KL-based update stop. It is a critic-free episodic policy
gradient variant, not a claim of a new GRPO theorem or full actor-critic PPO.

Attention/head learning rates are 1e-5/5e-5 for supervised learning, then
1e-6/1e-5 for RL. Gradient accumulation four and clipping one preserve the real
FP32 native LoRA path. Episode reward gives coarse credit to all three decisions;
this is a known limitation, not precise per-trade causal attribution.

### 3. Model inputs and outputs

The pinned original model sees every stock's identity, business, price, holdings,
current weight, several trailing returns, drawdown, volatility, volume ratio and
affordability, plus account capital/cash/equity and risk limits. Deterministic
UP/DOWN and volatility categories supplement the numerical values. They summarize
past observations and are not trading recommendations. Shuffled row IDs reduce
direct reliance on the ticker suffix at the output query.

Native attention and original-vocabulary LoRA emit five categorical allocation
actions. No external numerical policy head, cached trading sequence or heuristic
fallback chooses model actions. The five slots are conditionally independent
given context, not an autoregressive jointly conditioned allocation sequence.
Budget/risk projection can make different proposed actions execute identically.
Greedy evaluation versus stochastic training is an additional limitation.

## Evaluation contract and what can legitimately be claimed

Training: 2015–2023. Checkpoint validation: 2024, maximizing net growth at the
predeclared 100,000 INR primary capital, earliest ties; untrained and SFT-only
checkpoints remain eligible. Primary retrospective test: January 2025–13 November
2026. The historical and later slices remain supporting reports, not replacements
for a disappointing continuous result. No cherry-picked stock or starting capital
becomes the primary result after outcomes are seen.

Comparators include untrained, supervised-only, the causal teacher, equal-weight
holding, equal-weight rebalancing, inverse-volatility rebalancing, momentum,
cash and every individual-stock hold. Individual-stock holding is unconstrained,
so the maximum is a deliberately strong hindsight reference, not an investable
known-in-advance strategy. All receive matched dates/prices/cost assumptions.

Minimum desired result: a positive selected trained return beating raw and
equal-weight holding on the primary interval. Also report the stronger controls,
drawdown, volatility, Sharpe, fees and capital dependence. Show separately whether
RL improves upon supervised-only. If selection returns SFT or zero, do not label
it an RL success. If confidence is weak or performance regime-specific, say so.

No one can prove "no flaws" or "cannot get any better" from this finite search.
The honest claim is that specified methods were tried within a documented budget,
with the outcomes and known limitations retained. Test periods have been viewed
repeatedly and informed method revisions: this is adaptive retrospective research,
not an untouched final confirmation. Publication claims must reflect that.

## Starting money: correct interpretation

Whole-share quantities, unspent cash and affordability should change with capital.
The model also receives capital and holdings; explicit counterfactual probes test
its output sensitivity separately. Different executed quantities alone do not
prove that the adapter learned capital-dependent preferences. At larger balances,
similar percentage returns and allocations can be economically reasonable.
No objective rewards differences merely to make the demo look adaptive.

## Preflight, artifacts and operation

17 tests passed: execution conservation, costs, whole-share funding, split handling,
hold persistence, no future observation dependence, concentration rules, query
mapping after permutation, teacher mapping and risk response, group advantage
behavior, likelihood ratio/gradient and training-window boundaries.

Actual GPU preflight collected four account trajectories and made six optimizer
steps. Nonzero parameter change 0.0000584581; rollout/update probability error
0; export/reload error 0; adapter-disabled recovery error 0; frozen-reference
swap error 0. Peak allocated GPU memory 5,885,360,128 bytes; at most 961 tokens in
this probe. This establishes machinery operation, not profitable learning.

Source: `research/stateful_rl_20261117_run3/`.
Run: `C:/Users/Parth/Documents/PolyvalentFinance/campaign_runs/stateful_curriculum_episode_seed42_20261117`.
Logs: `PolyvalentFinance/reports/stateful_curriculum_episode.stdout.log` and stderr.
Launch: 17 November 2026 14:46:40 IST; wrapper PID63884, actual PID in status.
No automatic promotion to the menu. Do not reopen the old GPU demo while training.
No quiet-hours or wall-clock stop is encoded. Temporary process-only sleep
inhibition ends with the worker. Check status/logs before starting anything else.

The worker saves the protocol/source hashes, curriculum, checkpoint metrics,
selected artifacts, account-trajectory receipts, state counterfactual diagnostics,
all baseline ledgers and comparisons. The original data reconstruction and vendor
corporate-action assumptions remain; taxes, liquidity and market impact are not
fully modeled. Five contemporary selected stocks are not a broad market claim.
