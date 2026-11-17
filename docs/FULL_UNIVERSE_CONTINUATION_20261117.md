# Full-universe continuation: 17 November 2026

This chat recovered the opening discussion, the user's "astra my man" reset,
and every subsequent turn through the unfinished all49 rebuild. The authoritative
scope is the original InternVL3-2B, original Nifty CSV and every one of its49 stocks.
Earlier5-stock results cannot satisfy that scope. The old static-mixture model
never received changing portfolio information and is historical only.

## Active experiment

Run: `C:/Users/Parth/Documents/PolyvalentFinance/runs/all49_20261117_b`.
Worker: `research/full_universe_rl/train.py`. Read status.json and verify its PID;
do not infer that training is running from this document. Stdout/stderr are beside
the run directory. Source snapshot, fixed protocol, data audit, optimizer/RNG
recovery and native LoRA+input-bridge checkpoints are stored with the run.

Initial launch a completed its first update, then was stopped immediately to fix
target sizing during outstanding corporate-action receivables: pending value
must count toward target equity but cannot fund purchases. Run b starts cleanly;
no checkpoint or return was selected from a. Fifteen automated checks passed
before launch b. These tests reduce specific risks; they do not prove perfection.

## What is implemented

Original CSV SHA256:29b847a33ad88b7568eef66e1fb9f3274a897bd3091ddb8551c129b75dd726f7.
No source rows or tickers removed/replaced. The executable experiment begins
July2018 because all49 stocks then have126 genuine historical observations.
1999-2018 source history remains retained; no pre-listing history is fabricated.
SFT:July2018-Dec2022. Learning gate:2023. Financial RL:July2018-Dec2023.
Financial checkpoint validation:2024. Primary retrospective test:Jan2025-Jan2026,
exactly within the original dataset. No silent extension to November2026.

All49 stock numerical records precede all49 action queries. Each query also
receives its own numerical state, eliminating reliance solely on learning the
association between49 earlier numerical tokens and ticker text. Each record
contains prices, holdings, values, weights, affordability, returns, risk, volume,
and account information. All choices are original InternVL vocabulary logits.
The20->128->1536 input bridge is an added trainable representation, not an external
trading policy. It is saved and restored together with native rank16 LoRA.

The matched untrained initialization has a zero-final-layer input bridge. Thus
trained-minus-untrained measures combined representation and LoRA training;
it cannot by itself establish the isolated value of the language model or LoRA.
Numerical-only and bridge/LoRA ablations remain necessary for publication claims.

## Execution and disclosure

Original adjusted prices supply signals. Captured supplementary OHLC supply
execution prices after undoing future split/bonus unit adjustments. Reliance2023
and ITC2025 pre-demerger quotes are reconstructed using published pre-event closes.
The exact inputs/factors/hashes/sources are in execution_audit.json. These are
reconstructed vendor prices, not a fully exchange-certified price database.

Demerged shares become non-spendable receivables, marked at the published discovery
value until listing; the simulator liquidates them mechanically at listing open
with fees/slippage. This keeps the original investable universe intact without
losing entitlements or inventing immediately available cash. Fractional bonus and
child entitlements use a disclosed cash-in-lieu approximation; exact trustee
settlement dates/proceeds are not modeled. Dividends are credited on ex-date.
No tax, liquidity constraint, delayed settlement or impact model is claimed.

Actions target0/.5/1/2/4/7/10percent of equity; combined weights normalize above100.
A10percent requested stock cap is fixed across budgets. Unaffordable shares remain
cash. At500INR this can produce no trades; that is the declared diversification
constraint, not proof of a learned strategy. Account-size response must be measured
in both raw actions and actual affordable fills, never manufactured.

## Learning and evaluation

Causal inverse-volatility initialization includes feasibility checks and25percent
synthetic-risk augmentation. Loss is class-weighted. Every training example
contains and predicts all49 assets. Check the real2023 learning gate every128
examples; require balanced accuracy>=.60 and at least.15 above initialization,
minimum eligible class recall>=.20, and at least4 predicted classes. Failure stops
before financial RL. This gate establishes heuristic learning, not an investment edge.

Financial RL: four own-account60-session trajectories per context, two causal
regime-balanced contexts, three20-session actions; factorized clipped probability
ratios share trajectory advantage. Reward is net log growth minus0.1drawdown.
Keep SFT and RL separate. Select by2024 net return with earliest ties, permitting
untrained/SFT to win. Evaluate matched trained, untrained, SFT, teacher, holding,
equal-weight rebalancing, inverse volatility, momentum and cash.

Primary test budgets:500/5000/50000/100000/200000INR. Separately initialized2025
and January2026 diagnostics at100000INR. Same accounting for all policies.
Repeated prior inspection means these are retrospective tests, not an untouched
holdout. Fixed contemporary stock membership creates survivorship bias. No claim
of the globally best approach, guaranteed outperformance or a flawless process.

## Next check and recovery

Confirm actual optimizer updates in status/stdout and update_reload_receipt.json.
Inspect gate_history.json; if failed, diagnose before another run. If passed, the
worker continues through96 RL iterations and evaluation automatically.
`recovery.pt` saves parameters, optimizer, Torch/CUDA/NumPy/Python RNG, sampler RNG
and frozen SFT reference. Saves every16 supervised updates and every RL iteration.
After a crash, use the same command plus `--resume`; source hash changes are rejected.
A restart may replay at most15 supervised updates or the unfinished RL iteration.

The literature supports the staged idea, not our hoped-for performance:
https://arxiv.org/abs/2608.09880
https://arxiv.org/abs/1707.06347
https://arxiv.org/abs/2402.03300

## Verified early progress

Actual workerPID21336 completed128 supervised updates. First-update probability
change0.0334439874; exact checkpoint reload error0; exact disabled recovery error0;
peak allocated GPU memory5,816,025,600bytes. Full recovery snapshots are present.
At128updates,2023 gate balanced accuracy22.28percent: not passed yet, and not a
financial result. Continue the declared curriculum; do not lower the gate to
claim success. Loss fell from3.263 on the first example to1.470 over the latest16.

An independently trained16519-parameter numeric MLP control (exact count in
numeric_control_summary.json) attained92.23percent balanced gate accuracy on the
same target examples. It uses own numerical features and pooled mean/std context;
no language model or RL. This establishes learnable targets, not InternVL benefit.
Its2024 validation return was5.582percent. Full49 fixed baselines at100000INR in
2024: holding6.930percent; equal rebalanced6.009percent; inverse volatility5.834
percent; top10 trailing120-session momentum29.506percent. No test-period results
were examined for this continuation. These are protocol-specific benchmarks,
not interchangeable with earlier5-stock or fractional-share numbers.

A coarse execution reconciliation found no economic daily-return difference above
3percentage points versus original adjusted signal returns across the experiment
period, including discovery-valued demerger entitlements. This is a diagnostic,
not proof of fully accurate vendor quotes or exact corporate-action settlements.

Code and initial receipts pushed as c611d46 under Parth Dhande. Local full source
snapshot and optimizer recovery remain in the run directory. Training continues
independently; inspect live status to see whether the gate passes or fails.
