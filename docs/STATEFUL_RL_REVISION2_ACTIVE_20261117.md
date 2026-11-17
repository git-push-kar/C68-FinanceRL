# Active state-aware Finance RL revision 2

17 November 2026. **No conclusive winning result has been established.**

The user clarified that several justified attempts are permitted within the
original 12–24-hour total budget. The earlier artificial two-run ceiling is
superseded. Do not use this as authorization for an open-ended search, altered
test periods or a return to the constant-mixture adapter as the final answer.

## First full-state attempt: stopped, not promoted

Run: `PolyvalentFinance/campaign_runs/stateful_full_account_seed42_20261117`.
The actual model saw the stock/account observations and had real native LoRA
gradients. However, 2022 validation at 100,000 INR fell from the untrained
**+1.432%** to **−8.106%** at epoch 1 and **−18.308%** at epochs 5–8. At epoch 8,
all 50 validation dates had the same action vector, concentrated in Infosys.
The process was stopped during epoch 10, preserving its original nominal
12-epoch protocol and a separate manual-stop explanation. No test result was
used to decide this revision. This is a failed learning/generalization result,
not evidence that the model's inputs were absent or that the hardware caused it.

The validation records are consistent with a stock-identity shortcut and excessive
concentration; they do not isolate one universal cause. The raw model's low
exposure also means its positive validation result is not automatically a strong
investment strategy. Failed-run evidence is under `stateful_rl_run1_failure/`.

## Revised method

Source: [stateful_rl_20261117_run2](../research/stateful_rl_20261117_run2/README.md).
The pinned 2B model, native attention/vocabulary LoRA, five stocks, full account
inputs and whole-share next-open accounting are retained. Main changes:

- Training 2015–2023; validation 2024; the primary test remains continuous
  January 2025–13 November 2026. The old validation year is explicitly training
  data now; there is no claim of independent untouched confirmation.
- Twenty-session decisions and rewards, with longer momentum/drawdown inputs,
  rather than the old five-session objective.
- Explicit rebalance concentration limits (normally 35% per stock, 60% per sector)
  with documented small-account feasibility relaxation. The model sees the limits.
- Full stock rows/query order shuffled, with canonical action mapping verified;
  identities, prices, positions and account facts are never hidden.
- Smaller learning rates, stronger KL regularization, early checkpoint checks and
  validation-based early stopping. Selection still maximizes absolute validation
  wealth growth at 100,000 INR, not Sharpe or a favourable trained/raw delta.

These changes are made together under a restricted engineering budget. Do not
claim that this experiment isolates each one's causal benefit or creates a new
policy-gradient theorem. Details, limits and stronger holding controls are in the
method README. All-action one-decision-horizon contextual financial RL is the
actual algorithm; it must not be labelled PPO or optimal long-horizon control.

An additional tokenizer test found that a punctuation/newline combination may
share one token. The old endpoint rule selected the preceding name token rather
than the token containing the full query endpoint. The rule was corrected and
canonical mapping tested explicitly. A setup launch was aborted during pristine
reference collection (64 examples, no optimization) because its shell launcher
had not stopped on the failed test's exit status. That launcher gating was also
corrected. The aborted setup remains in its original directory with a record;
the active run uses a fresh directory and passed all gates. This issue is not
asserted as the sole cause of the first full run's failure.

## Active worker

- Actual run: `C:/Users/Parth/Documents/PolyvalentFinance/campaign_runs/stateful_monthly_risk_seed42_20261117_v2`.
- Source: `C:/Users/Parth/Documents/C68-FinanceRL/research/stateful_rl_20261117_run2`.
- Logs: `C:/Users/Parth/Documents/PolyvalentFinance/reports/stateful_monthly_risk_v2.stdout.log` and `.stderr.log`.
- Seed 42; maximum 8 epochs, 1,536 training states, accumulation 4.
- Checkpoints at examples 32/64/128, then every 256. Epoch zero eligible; earliest
  tie. Early stop after 24 checks without validation improvement. No clock cap.
- Automatic selected-weight reload, pristine recovery, state probes and full
  trained/untrained/holding/stronger-control evaluation follow training.
- No automatic promotion to the menu. The old GPU demo remains stopped.

Expected runtime is roughly 3–4 hours from this launch, possibly shorter if the
declared validation early-stop rule fires. Verify actual process/status before
operating; do not duplicate the worker or run another GPU model alongside it.
The 12–24-hour user budget governs human/agent decisions, not encoded time checks.

## On the next check

Read `status.json`, logs, `candidates.json` and `selection.json` for this **v2**
run, not the stopped first attempt or aborted preflight directory. Review the
primary continuous test at the declared capital, all budget stress results and
the actual action records. Useful response to stock/account state and improved
matched performance are both needed. A changed probability, nonzero gradient,
training utility gain, favourable slice or weaker raw baseline alone is not enough.
