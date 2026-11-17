# State-aware financial RL revision 2 — 17 November 2026

This is a response to a recorded learning failure, not a claimed winning result.
The first full-state weekly run was stopped during epoch 10: every completed
trained checkpoint was below its initial validation return, and epoch 8 chose
the same Infosys-concentrated action on all 50 validation dates. State inputs and
native LoRA gradients were real; useful state-dependent generalization failed.
The original failed run, protocol, weights and manual-stop record are preserved.

## What remains fixed

Same pinned InternVL3-2B, full native model and rank-16 attention/vocabulary-head
LoRA (6,809,504 trainable parameters). Same five known stocks, captured raw prices,
split/dividend reconstruction, whole shares, next-open chronology, cash funding,
10 bp fees plus 5 bp slippage per side. Every observation contains stock names,
businesses, prices, current quantities and weights, recent returns, volatility,
volume, cash, current equity, initial capital and cash-based affordable quantities.
There is no learned numerical substitute, stored-action fallback, or static
four-strategy mixture. Market and account facts enter the actual model forward.

## Changes and reasons

1. **Use available pre-test history:** training now covers 2015–2023, validation
   2024, test still begins 2025. The former 2022 validation is explicitly training
   data now. This includes more recent contrasting market conditions instead of
   leaving 2023–2024 unused. It does not create an untouched confirmation study.
2. **Twenty-session decision/reward horizon:** use a slower portfolio-allocation
   cadence with fewer transactions, rather than optimizing noisy five-session
   outcomes. Add 120-session returns and 60-session stock drawdowns to the input.
   The model receives that cadence; execution and financial rewards use it too.
3. **Explicit concentration constraints:** normally at most 35% per stock and
   60% per sector at rebalancing. If only one or two stocks/sectors can be afforded
   using total account equity, caps relax to permit a feasible small account.
   These rules are in the observation. They override KEEP at order execution if
   necessary. The model still chooses per-stock targets and may retain cash.
   Limits are applied when trading, not falsely claimed as continuous intraday
   risk enforcement. An existing position may drift between rebalances.
4. **Reduce identity/position shortcuts:** shuffle complete stock rows and query
   order during training while preserving every stock's actual identity and data.
   Gather output slots back into the canonical asset order before financial
   evaluation. This is augmentation, not proof of permutation invariance.
5. **More conservative updates and earlier validation:** attention/head LRs are
   1e-6/1e-5. KL coefficient is 0.1, rather than 0.02. Save and validate examples
   32/64/128, then every 256, rather than waiting a whole epoch. Zero remains
   eligible. Stop after 24 validation checks without improvement; this is an
   overfitting rule, not a wall-clock or quiet-window restriction.

The action letters are KEEP, sell all, target 5/10/20/30%, or the current maximum
permitted stock weight. Excess combined targets are normalized, with integer
concentration/funding checks at the next open. This avoids several redundant
100%-style requests all collapsing onto one risk-capped action.

Training is exact all-action, one-decision-horizon contextual financial RL over
7^5 joint actions. Utility is net log wealth gain minus 0.1 maximum horizon
drawdown; the financial term is scaled by 0.05 and regularized to pristine-model
probabilities on the same augmented observation. It is not teacher SFT, PPO or
a proof of long-horizon optimality. The established policy-gradient/all-action
foundations are linked in the [original method README](../stateful_rl_20261117/README.md).
The risk caps and reward coefficients are explicit study design choices, not a
new theoretical result or parameters selected on test profits. Multiple changes
are made together; this run cannot isolate each component's causal contribution.

## Selection, controls and gates

Primary checkpoint criterion: absolute 2024 net wealth growth at 100,000 INR,
earliest tie, zero eligible. Sharpe is reported but does not choose the checkpoint.
The primary test remains continuous January 2025–13 November 2026; other slices
are supporting results, not alternative convenient headlines. Test capitals are
500/50,000/100,000/200,000 INR.

Trained and disabled-adapter untrained share every active execution/risk rule.
Equal buy-and-hold buys once and genuinely holds, without forced risk rebalancing
afterward. Single-stock holding benchmarks are intentionally unconstrained so
their initial 100% purchase is not weakened into a 35% position; their hindsight
maximum is labelled a stronger, retrospectively chosen comparator. Other controls
are rebalanced equal weight, inverse volatility, momentum and cash.

The model must show useful action variation and actual portfolio-aware execution,
not merely a nonzero gradient, changed probabilities or a higher training reward.
Report failure if it does not improve the matched evaluation. No automatic menu
promotion occurs. Names/cash/metrics remain visible; they are never removed to
make a benchmark artificially easy. The original five-stock data and all stated
dividend, impact, fixed-universe and foundation-pretraining limitations still apply.

The user permits several justified attempts within the original 12–24-hour total
budget; an earlier two-run limit has been explicitly lifted. No such usage or
clock cap is coded into the method.
