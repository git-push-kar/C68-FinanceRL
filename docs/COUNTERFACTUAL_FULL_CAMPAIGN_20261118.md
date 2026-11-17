# Counterfactual all49 financial RL: complete campaign

18 November2026. User authorized a materially different, full-fledged approach,
not a short trial followed by another planning decision. Three fixed seeds42,
123,456 run1024 supervised updates and1536 financial decisions each, contingent
only on predeclared correctness/learning gates. No promise of profitability.

## Why change the method

Historical static-mixture policy never saw changing stock/account state. Later
five-stock experiments changed the required universe and cannot conclude this
domain. The first restored all49 run learned conditional decisions, but its best
2024 RL checkpoint returned3.6663percent on2025-Jan2026, versus3.5905percent raw,
3.6711percent SFT and6.3586percent holding at100000INR. More RL steps did not help.
Shared trajectory rewards assigned the same advantage to49 distinct decisions;
this makes attribution noisy. Row permutation changed actions. Independent small
weight bins and floor-only whole-share allocation left substantial idle cash.
These are observed design weaknesses; their causal contributions are not isolated.

## The complete new method

Keep pinned original InternVL3-2B, all49 original dataset stocks and unchanged
original adjusted signal data. Relative score actions0/.25/.5/1/2/4/8 replace
absolute weight bins. Positive scores normalize with a10percent per-stock cap;
fewer than10 active stocks imply deliberate residual cash. Model chooses every
stock score, with no hidden teacher fallback. All zero means cash.

The25 input features contain the previous20 portfolio/market fields plus causal
cross-sectional ranks of20/60/120-session returns and volatility, and price/equity.
These engineered features expose contemporaneous comparisons, not future returns.
All49 individual records remain present; the25->128->1536 trainable input bridge
feeds the unchanged28-layer native decoder, and rank16 attention/native-vocabulary
LoRA produces the actions. Canonical ticker sorting makes caller row order
irrelevant. This is preprocessing invariance, not proof of an intrinsically
permutation-equivariant neural architecture or absence of fixed-position bias.

Supervised initialization teaches a causal relative-score rule. Three teachers
(inverse volatility, momentum ranks, risk-adjusted momentum) were compared only
on2019-2022 mean annual log growth minus0.1drawdown. Momentum won0.159451 vs0.131249
and0.142057. It supplies training labels and remains a separately reported
benchmark. It never supplies the learned policy's inference actions.

SFT usesJuly2018-2021 observations, including25percent synthetic risk/trend
augmentation;1024 full49-target examples.2022 learning gate checks at256-step
intervals. Select qualifying SFT by allocation error, not financial return.
Require balanced accuracy>=.60, minimum eligible recall>=.20, improvement overraw
>=.15 and allocation L1 error less than80percent of uniform baseline error.
Failing seeds are recorded as failures, not silently extended or relabeled RL.

RL runsJuly2018-2022 only. Each account takes3 consecutive20-session decisions.
At each decision sample all49 actions, then hold48 fixed while enumerating all7
choices for the remaining stock.343 alternatives supply exact simulator utilities:
20-session net log growth minus0.1drawdown. The categorical conditional expected
utility uses a per-stock probability-weighted baseline, fixed reward scale0.01
andKL0.02 to the frozen SFT policy. A single optimizer update follows each reward
matrix. There is no learned critic and no shared global advantage for all49.

This is receding-horizon, simulator-based financial RL. It does not optimize an
exact long-horizon value function; future consequences beyond20 sessions are not
credited directly. Stochastic training vs greedy categorical evaluation remains
a disclosed approximation. No claim to reproduce COMA or invent a new general RL
algorithm. Lower learning rates and frozenSFT regularization constrain drift.

## Execution revision applies to every policy

Signals remain original adjusted prices; executions use the previously audited
supplementary reconstructed raw OHLC and corporate-action ledger. No stock dropped.
The allocator floors targets, then buys additional affordable whole lots only
when they reduce squared stock-plus-cash allocation error. Zero-target assets
are never purchased; sell-before-buy, fees,slippage and10percent post-cost cap
are enforced. Target cash is respected. Canonical index breaks exact ties.
This is a deterministic execution optimizer, not a learned alpha/trading policy.

The change is materially significant: at100000INR, early holding cash fractions
in2023 and2024 are0.116percent and0.139percent, respectively. Corrected holding
validation returns are30.999percent(2023) and13.623percent(2024). These new matched
benchmarks are NOT interchangeable with old floor-only or fractional-share runs.
All original corporate-action simplifications, fixed-universe survivorship bias,
missing liquidity/tax/settlement models and reconstructed-price limitations remain.

## Selection, final comparisons and controls

RL selection uses mean annual net log growth across2023 and2024, checked every128
updates through1536. Untrained and SFT remain eligible; earliest tie wins. All3
seeds reported, no best-test-seed cherry-picking. Test access is deferred until
all3 native seeds finish training. Primary period remainsJan2025-Jan2026 at
100000INR; budget diagnostics500/5000/50000/200000; separate2025 andJan2026 slices.
Repeated prior access means this is adaptive retrospective research, not pristine
confirmation. No parameter changes may be justified by a desire to rescue that test.

Each seed compares selected,untrained,SFT,teacher,holding,equal rebalanced,
inversevol,momentum,cash. Selected checkpoints are reloaded and must reproduce
validation equity exactly. Probe capital,holdings,returns,risk and reorderedrows.
Ablate the numerical input bridge and nativeLoRA separately. Include doubled costs
and descriptive20-session block-bootstrap intervals; these do not correct for
all prior adaptive experiment selection. No automatic demo promotion.

A separate18439-parameter CPU sharedMLP uses the same25 own features plus pooled
cross-stock means/std, same SFT states, counterfactual RL, seeds, update counts,
execution and validation. Learning rates are model-specific, so optimization is
not identical. Other-stock context is pooled, whereas InternVL sees all individual
records; control does not use ticker identity. Feature parity is verified exactly.
This control is not the requested2B model and cannot substitute for its results.
Its test is also deferred until all native training has finished.

## Checks and current state

11 tests passed before launch: money conservation, no borrowing/shorts, corporate
entitlement persistence, fractional bonus handling, causal inputs, deterministic
holding, original quote anchors, capped score projection, funded residual lots,
canonical input identity, reward-gradient direction and training time boundaries.
Actual first GPU update changed probabilities by0.05786508; reload error0;
disabled-adapter recovery error0; peak allocated GPU memory5,837,867,520bytes.
These establish specified mechanics, not financial success or absence of all bugs.

Main run: C:/Users/Parth/Documents/PolyvalentFinance/runs/counterfactual49_20261118
Main source: research/counterfactual_rl; launch with python -m research.counterfactual_rl.train.
Control source: research/counterfactual_numeric_control.py.
Read status.json and process state; this document is not proof a worker remains live.
Native recovery saves model, optimizer, RNG, SFT reference and evolving account
every32SFT or16RL updates. Resume same command with --resume; changed source hashes
are rejected. Numeric control has per-seed recovery snapshots; no automatic resume
loader is yet supplied. Parent campaign and numerical worker run independently.

## Primary sources informing this design

- Counterfactual credit assignment: https://arxiv.org/abs/1705.08926
  Original uses a learned centralized critic for cooperative agents. Our adaptation
  uses exact simulator rewards over stock coordinates and a finite holding horizon.
- Portfolio/resource permutation structure: https://arxiv.org/abs/2102.09361
  Motivates respecting symmetries; canonicalization here is a narrower remedy.
- Shared portfolio evaluators/state: https://arxiv.org/abs/1706.10059
  Their crypto results are not comparable to this equity experiment.
- Native token allocation and staged learning: https://arxiv.org/abs/2608.09880
  Their causal mean-variance SFT/GRPO ETF experiment motivates staged learning,
  not a guarantee of our scores, method novelty or paper acceptance.
