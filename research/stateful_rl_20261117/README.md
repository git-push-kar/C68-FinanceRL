# State-aware native Finance RL attempt — 17 November 2026

Purpose: replace the old constant-prompt strategy mixture with actual per-stock
actions conditioned on market and portfolio observations. This attempt is not
declared successful merely because training runs or a model call is genuine.

## Decision and execution contract

The full pinned InternVL3-2B base is preserved. Only its rank-16 attention and
original vocabulary-head LoRA are trained (6,809,504 parameters). There is no
separate numerical policy making the decisions. One forward pass sees all five
stocks, closing prices, 5/20/60-session returns, 20-session volatility, relative
volume, held quantities, current weights, affordable shares, cash, current equity
and initial capital. No input is silently truncated. Seven original-vocabulary
action logits are read at each of five stock-specific assistant allocation slots.

The five-stock universe is Coal India, HDFC Bank, Infosys, SBI and TCS. This is a
small, concentrated financial-management study, not a whole-Nifty performance
claim. Reliance was excluded before policy evaluation because its Jio demerger
requires entitlement accounting not captured by ordinary cash dividends/splits.
The initial unselected raw capture remains in `data/`; final data is `data_v2/`.
Universe selection was not based on measured model/test returns, but fixed-universe
survivorship and concentration remain limitations.

Each stock action is KEEP current shares, sell all, or request 10/20/40/60/100%
of current equity. Excess combined targets are normalized. These are per-stock
actions, not a fixed blend of investment strategies. The executor applies whole
shares, no shorting, no borrowing, sells before buys and proportional cash funding.
KEEP preserves quantities rather than accidentally rebalancing overnight gaps.
The model observes one known close and trades at the next open. It cannot see
that opening price. Reconsideration occurs every five sessions.

Costs are 10 bp fees plus 5 bp adverse slippage per side. Dividends accrue to
overnight holders on recorded ex-date (not actual payment date), and integer
splits adjust share quantities before opening trades. Source split-adjusted OHLC
and dividends are reconstructed to historical share units. The incomplete latest
Yahoo session is omitted without filling its close. These assumptions are shared
by every comparator; this is not broker-grade tax/impact/payment-date simulation.

## Financial learning objective and scope

For a valid training observation, the simulator evaluates all 7^5 joint actions
over the following five sessions, including funding, corporate actions and costs.
The loss maximizes the exact expected net log wealth gain under the model's
factorized action distribution, with a fixed KL regularizer to the original model.
Only the probabilities are differentiated. Future prices are rewards, never
observations; there are no teacher answers or synthetic explanation targets.

This is **offline contextual financial RL / one-rebalance-horizon all-action
policy gradient**. It is not PPO or a proof of optimal long-horizon portfolio
control. Training states contain randomized valid inventories and budgets, rather
than pretending to be entirely on-policy long episodes. Sequential evaluation
feeds the resulting actual positions/cash back to the model every decision.
Training expected-action utility and evaluation per-stock argmax are explicitly
different, as in ordinary stochastic-policy training with greedy evaluation.

Foundations, not novelty or directly matched external return benchmarks:

- [All-Action Policy Gradient Methods](https://arxiv.org/abs/1910.09093): integration
  over actions rather than relying only on sampled action gradients. Our finite
  product action space permits exact enumeration; their experiments are not ours.
- [Policy Gradient Methods with Function Approximation](https://proceedings.neurips.cc/paper_files/paper/1999/hash/464d828b85b0bed98e80ade0a5c43b0f-Abstract.html): established policy-gradient context.
- [FinRL environment](https://github.com/AI4Finance-Foundation/FinRL/blob/master/finrl/meta/env_stock_trading/env_stocktrading.py): explicit market/account state and funding constraints. This custom code is not advertised as a reproduction.
- [Yfinance split-price discussion](https://github.com/ranaroussi/yfinance/discussions/1682): historical Close remains split adjusted; source-unit reconstruction is recorded and audited, not confused with a contemporaneous tradeable share price.

## Locked selection and evaluation

Training rewards: 2015–2021, with no horizon crossing the training boundary.
Validation: 2022. Checkpoints are selected by absolute net wealth growth at
100,000 INR, earliest tie, with epoch zero eligible. Other capital sizes are
prespecified stress tests, not a route to selecting a convenient winning amount.

The primary test is the continuous January 2025–13 November 2026 path. The old
2025–January 2026 span and separately reset February–November 2026 extension are
supporting slices, not alternative headlines chosen after viewing results. These
are retrospective project evaluations; no blanket untouched-holdout claim is made.
The five-stock whole-share results cannot be directly ranked against the old
49-stock fractional 13.28% result.

Every test includes trained, actual disabled-adapter untrained, equal buy-and-hold,
equal rebalanced, inverse-volatility, momentum and cash controls. Each stock's
individual buy-and-hold is also reported; their hindsight maximum is labelled a
retrospective comparator, not a deployable foresight-free policy. All use the same
prices, dates, actions, funding and costs where applicable.

## Preflight and operation

Nine meaningful CPU/gradient tests check affordability, conservation, KEEP and
buy-and-hold, split quantities, vectorized action reward parity, future-input
invariance, required state fields and the exact expectation/gradient calculation.
GPU smoke checks verify complete inputs (708 tokens in the probes), nonzero
attention/readout gradients, a real optimizer update, pristine recovery and exact
export/reload (zero measured difference). Peak measured allocation was ~5.6 GB.

The training run is separate from those short preflight checks. One primary seed
and fixed settings are used; no wall-clock stop, quiet window or encoded usage
cap is part of the algorithm. The Windows worker temporarily inhibits automatic
system sleep while it runs; it does not keep the display awake or change a power
plan. Logs/status/exports are written independently of the Codex chat.

After completion, inspect comparison margins, actual action variation, account
sensitivity and fresh weight reload before promoting anything into the menu.
Failure to beat untrained/holding or failure to use state must be reported, not
hidden with a different capital, date slice, checkpoint or cosmetic demo.
