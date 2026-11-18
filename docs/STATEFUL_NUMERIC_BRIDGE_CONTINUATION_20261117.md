# Guarded continuation: numerical input representation

Status17November2026,17:42IST: prepared and CPU-tested, **not yet GPU-tested or
trained**. This is a conditional fallback, not an additional concurrent GPU run.
Current run4 and its criteria are documented in STATEFUL_RL_REVISION4_ACTIVE_20261117.md.

## Evidence motivating the fallback

Revision3 collapsed to constant20%actions. Revision4's focused text representation
improves its first held-out balanced-accuracy check from14.58% to39.58%, but still
has zero recall on some allocation classes. It continues its scheduled training;
that early result does not justify declaring all text conditioning impossible.

A separate CPU control used exactly the current training/gate examples with
continuous numeric features and a38->64->64->7 classifier. At the fixed final
200epochs it scored about88% on the held-out gate; earlier checks were similar.
This is evidence that the causal allocation labels are learnable from the state,
not that a profitable trading signal exists. It is neither InternVL nor RL and
will not replace the portfolio policy. Artifacts:
`stateful_rl_revision4_preflight/numeric_learning_control.json`.

## What changes in revision5

The unchanged pinned InternVL3-2B decoder and original vocabulary projection
remain the action path. Add a small numerical **input encoder**:

-38 continuous causal features: focus stock returns/risk/price/holdings,
  cash/equity/initial-capital facts, limits and other-stock context;
-38->128 linear,tanh,128->1536 linear,then0.02*tanh bounded residual;
-add this residual to the final action-query **input embedding**, before all
  28original decoder layers, not to the final logits;
-all original text stock/account facts remain present;
-201,600 additional trained input-adapter parameters, alongside existing
  6,809,504attention/vocabulary LoRA parameters. No external action classifier.

The input encoder's final layer starts at zero, preserving the initial model
output. Turning the adapter off bypasses the encoder as well as LoRA, providing
a matched pristine baseline. Fixed scaling/clipping uses no future-fit statistics.
Other-stock numeric rows are ordered by known volatility; no ticker-ID feature
is in the numeric encoder. The text still supplies all identities.

This is an explicit architecture change. A paper must disclose and ablate the
input encoder; it cannot claim that unmodified text-only LoRA solved numerical
conditioning. Background includes [Time-LLM](https://arxiv.org/abs/2310.01728)
and [numerical feature embeddings](https://arxiv.org/abs/2203.05556). Those works
motivate representation adaptation, not our trading performance or a guaranteed
causal benefit of LLM size. Learned input adaptation can also weaken claims that
language pretraining is necessary; a future numerical-only trading baseline
would be needed to isolate that contribution.

Supervised and RL encoder learning rates are0.001 and0.0001. Remaining focused
curriculum/gate/RL protocols are inherited. Encoder weights are saved separately
as numeric_bridge.safetensors and included in the artifact hash manifest. The
loader restores both components; frozen-reference snapshots include both. GPU
preflight must show a nonzero encoder update, exact output reload, reference
restoration and adapter-disabled pristine recovery before any training launch.

## Supervision and stop behavior

`research/Continue-FocusedFinance.ps1` runs in a hidden PowerShell process,
PID23132 at launch (verify before acting). It waits for actual run4 worker exit,
then rereads final status. It launches revision5 only if run4 explicitly reports
`stopped_learning_gate_failed`. Arbitrary crashes, financial losses or a completed
run do not automatically cause another attempt.

Before fallback launch it runs16CPU tests, then a fresh GPU preflight. Any failed
exit/receipt stops continuation. Existing fallback output directories cause an
error rather than overwrite or a duplicate worker. On a passing gate, RL runs;
on a failed gate, the worker stops without financial test evaluation. After a
completed financial run, descriptive uncertainty/action audit runs automatically.
No candidate is deployed to the menu automatically.

Supervisor status: `PolyvalentFinance/reports/focused_continuation_status.json`.
Supervisor logs: `focused_continuation.stdout.log` and stderr in that directory.
Fallback run: `campaign_runs/stateful_numeric_bridge_seed42_20261117`.
Fallback preflight: `campaign_runs/stateful_numeric_bridge_preflight_20261117`.
These directories do not exist until conditional launch. There is no encoded
quiet period or wall-clock deadline; the15second supervisor poll is merely a
process/status check. It does not restrict active training.

## Required interpretation

Passing an allocation-imitation gate is necessary here, not sufficient for a
profitable finance RL result. Evaluate SFT separately from RL, preserve all
matched baselines, and do not switch the primary capital/period after losses.
Different share counts alone remain insufficient evidence of learned capital
adaptation. The test history is adaptively reused. A finite campaign cannot prove
an optimum or a flawless process. No successful final replacement is declared.
