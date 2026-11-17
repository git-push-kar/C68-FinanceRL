# Active state-aware Finance RL run — 17 November 2026

**This is a new, unreviewed training attempt. It is not yet the final Finance result.**

The user identified a scope failure: the old Nifty adapter learned a constant
strategy mixture without seeing stock prices, positions, cash or affordability.
Real model calls in its menu did not make that policy state-aware. The 13.28%
backtest remains a valid measurement of that narrower setup, but it does not
satisfy the requested portfolio-aware trading capability. Whole-share budget
constraints were also missing. Those limitations must not be disguised as a
model-size excuse or fixed through UI wording alone.

The user authorized a 12–24-hour total attempt, one or at most two substantive
runs, and asked Codex to become inactive once training is independently running.
No wall-clock cap or quiet-window restriction is encoded in the new algorithm.

## Primary run

- Source: `research/stateful_rl_20261117/`.
- Method, assumptions and primary sources: [method README](../research/stateful_rl_20261117/README.md).
- Run output: `C:/Users/Parth/Documents/PolyvalentFinance/campaign_runs/stateful_full_account_seed42_20261117`.
- Standard output: `C:/Users/Parth/Documents/PolyvalentFinance/reports/stateful_full_account_primary.stdout.log`.
- Standard error: corresponding `stateful_full_account_primary.stderr.log`.
- Seed 42, 12 epochs, 1,536 valid training states, gradient accumulation 4.
- Rank-16 attention + native vocabulary-head LoRA, 6,809,504 trainable parameters.
- Attention LR 3e-6; vocabulary-head LR 3e-5; clipping 1; zero weight decay;
  financial loss plus fixed KL to the pristine model on identical observations.
- Training rewards 2015–2021. Validation 2022 selects absolute net wealth growth
  at 100,000 INR, earliest tie, epoch zero eligible. A zero-epoch selection is not
  evidence that training helped.
- One primary continuous retrospective test: January 2025–13 November 2026.
  Earlier-span and later-extension slices are supporting evaluations, not alternate
  headlines selected for a favourable score.
- Fixed evaluation capitals: 500, 50,000, 100,000 and 200,000 INR.
- Actual adapter-disabled untrained comparison, equal buy-and-hold, rebalanced
  equal weight, inverse volatility, momentum, cash and all five single-stock
  holding comparisons. The hindsight best single holding is disclosed separately.

The adapter receives the full five-stock state and makes per-stock actions. The
executor implements integer shares, no borrowing/shorts, cash funding, next-open
execution, fees/slippage and specified split/dividend accounting. Future prices
only determine financial rewards, not observations. No cached action fallback or
constant-mixture fingerprint restriction is used. The algorithm is explicitly
one-rebalance-horizon offline contextual financial RL, not falsely labelled PPO
or an optimal long-horizon controller.

## Preflight evidence

`docs/stateful_rl_preflight/` contains actual prompt examples and GPU checks.
Nine CPU/gradient tests pass. Whole-share affordability, execution conservation,
buy-and-hold/KEEP quantities, split handling, vectorized reward parity and future
input invariance are covered. The exact finite-action expectation and gradient
match an independent indexed summation.

GPU preflight saw complete 708-token observations, nonzero attention and native
vocabulary LoRA gradients, a real parameter update, zero measured export/reload
difference and zero disabled-adapter recovery difference. A consistent change from
500 to 200,000 INR changed action probabilities by up to 0.066 in the selected
preflight probes. This demonstrates input sensitivity, not superior decisions.
Peak measured allocation was 5,624,967,168 GPU bytes. Three full gradient probes
including an uncached reference forward took approximately 0.82–0.90 seconds each.

Measured runtime suggests approximately 4–5 hours for the fixed run and automatic
comparisons; this is an estimate rather than a stopping condition or guarantee.
The old demo server was stopped to free GPU/RAM. The worker runs detached with
logs and atomic status updates and temporarily prevents automatic system sleep.
It must not be duplicated or compete with another GPU model process.

## On resuming Codex

1. Read `status.json` and both logs; determine whether training finished or failed.
   Do not launch another run merely because the chat was inactive.
2. Inspect validation `candidates.json`, `selection.json`, actual saved-weight
   reload and disabled-adapter recovery checks.
3. Read `comparison_summary.json`, `test_metrics.json` and full decision records.
   Check actual action variation, sensitivity, cash/positions and trade feasibility.
4. Judge the primary full test at the declared capital. Show all stress amounts
   and controls, including losses; do not pick a capital/period to manufacture a win.
5. Promote to the menu only after capability, accounting and performance are
   reviewed. No automatic deployment is performed by the worker. If a second run
   is warranted, it needs a specific diagnosis and must fit the remaining total
   time budget. No broader search campaign is authorized by this deadline.

No outcome, baseline superiority, journal acceptance, error-free optimality or
exhaustion of 2B potential is claimed before these results exist.
