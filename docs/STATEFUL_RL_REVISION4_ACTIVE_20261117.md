# Focused stock decisions and a conditional-learning gate

17 November 2026. Active experiment; no claimed investment success.

## Why this is a substantive next test

Revision3 supplied full state but selected the same20% allocation for every stock.
Its positive5000INR result exactly matched equal-weight rebalancing. This is not
evidence of learned state dependence or RL improvement. The completed failure is
preserved under `stateful_rl_run3_result` and must not be promoted.

Revision4 uses one focused stock per native InternVL decision. Every prompt still
contains all stock prices, holdings, market indicators and the total account.
The focus row and its cash/affordability/risk facts are explicit immediately before
the action query. The original model and vocabulary projection emit one action;
the same learned weights are shared across all five stocks. This addresses the
observed identical-answer failure without substituting a numerical trading model.
Five actual forwards produce a portfolio decision; the executor funds and caps it.

Shared per-asset evaluation has precedent in
[Jiang et al.](https://arxiv.org/abs/1706.10059), but this is not their architecture
or a reproduction of their crypto results. Numerical/table reasoning limitations
are also documented in [EMNLP2023](https://aclanthology.org/2023.findings-emnlp.1028/).
Neither citation guarantees that this particular representation will learn.

## Curriculum and mandatory gate

The teacher remains the causal inverse-volatility allocator selected in revision3
from training years only. Supervised states now use2015-2022; the learning gate
uses2023 real observations, with no shared decision dates. The gate year later
enters RL training explicitly; financial checkpoint validation remains2024.

Training samples are balanced across seven actions rather than dominated by20%.
The6000 candidate states include25% synthetic risk counterfactuals; balanced
resampling makes745/2048 actual training examples synthetic (36.38%), recorded in
dataset_summary. Only volatility measurements are perturbed; they teach conditional
allocation, not fictional profits. Prices/holdings/account values remain valid.
Gate examples are all unperturbed and drawn without replacement from their pools.

The gate contains96 examples:16each from six action classes. The5% class did not
occur in this2023 sample pool, so it is explicitly untested by the gate. A constant
answer scores16.67%; pristine model balanced accuracy is14.58%. Passing requires:

- balanced accuracy at least60% and at least15 percentage points above pristine;
- recall at least20% for every eligible class (at least8 gate examples);
- at least four distinct predicted actions.

These are predeclared engineering checks of conditional imitation, not a financial
benchmark or proof of investment skill. Check every512examples, up to2048. Stop
at the first passing checkpoint. If none passes, stop before RL or test backtests.
The selected SFT checkpoint is not chosen for a lucky financial return.

Native attention/head LoRA learning rates2e-5/1e-4 during this balanced warm-up.
Temperature1.0,5%uniform exploration,5%label smoothing. The legacy `--states`
argument remains in the inherited parser but is unused: learning_gate.py fixes
the2048-example pool. The manifest and actual dataset inventory are authoritative.

## RL only after the gate passes

Own-account60session episodes,three20session decisions,groups of four trajectories,
two contexts per collection. Account evolution uses actual model trades. Each
stock focus is updated separately with its categorical likelihood ratio clipped
to0.8–1.2 and the shared group-relative episode advantage. This is factorized
per-action clipping rather than revision3's joint five-action ratio.120focus
examples per collection,two update passes. Attention/head learning rates3e-6/3e-5;
KL/entropy coefficients unchanged. Reference is the gate-passing SFT model.

Reward remains net log wealth minus0.1 drawdown.96iterations maximum with
validation-based stopping after12checks without improvement. No wall-clock cap
or quiet-hours restriction. Predeclared2024 net-growth checkpoint selection and
all2025-2026 retrospective comparisons/capital probes remain unchanged. Earlier
test inspection makes method selection adaptive; no untouched-test claim.

## Independent CPU signal diagnostic

`research/stateful_signal_diagnostic.py` checks whether a simple numerical learner
can exploit the available indicators. Pooled ridge uses shared stock features,
no identity intercept, training-only standardization and20-session forward labels
whose outcomes finish before the next-year evaluation. It evaluates2022,2023,2024
only and does not inspect2025-2026. Three ridge strengths were recorded, not a
large test-period search. This is not an InternVL adapter or an RL result.

The result is mixed: some configurations beat holding in2022 or2024, but all three
lag substantially in2023. This does not establish that indicators are useless,
or that the language model is the sole problem. It argues against assuming a
strong, stable exploitable edge exists just because many indicators are provided.
Full metrics: `stateful_rl_revision4_preflight/causal_signal_diagnostic.json`.

## Checks and operation

15CPU tests pass, including complete focus mapping/account presence and rejection
of constant answers by the learning gate. Actual GPU preflight collected complete
portfolio trajectories and updated two focused optimizer batches. Maximum weight
change0.0000199863; likelihood replay, exported/reloaded output, disabled-adapter
recovery and frozen-reference swap errors all0. Peak GPU allocation5,778,608,128
bytes;757tokens maximum in the probe. These validate mechanics, not generalization.

Source: `research/stateful_rl_20261117_run4`.
Run: `C:/Users/Parth/Documents/PolyvalentFinance/campaign_runs/stateful_focused_gate_seed42_20261117`.
Logs: `PolyvalentFinance/reports/stateful_focused_gate.stdout.log` and stderr.
Started17:34:04IST on17November; actualPID15384, wrapper12160. Check status before
acting. No old demo server or second GPU worker should compete. No automatic
deployment. If gate fails, inspect class-level learning and representational
limitations before another financial RL attempt.

Potential next hypothesis, not yet implemented or launched: a learned numerical
input embedding bridge into the unchanged LM backbone, rather than forcing the
model to recover every quantity through decimal text. Relevant background:
[Time-LLM](https://arxiv.org/abs/2310.01728) and
[numerical feature embeddings](https://arxiv.org/abs/2203.05556).
Such a bridge would be an additional trained adapter component and must be
disclosed, saved, ablated and compared against its own matched initialization.
It must not secretly replace the LM with an external numerical action policy.
