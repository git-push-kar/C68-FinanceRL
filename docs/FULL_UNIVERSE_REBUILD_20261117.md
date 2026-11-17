> Update: implementation and launch are documented in FULL_UNIVERSE_CONTINUATION_20261117.md. Earlier pending-work statements below are historical; inspect current run status.

# Required full-universe rebuild — authoritative scope correction

The user explicitly rejected the five-stock reduction. Non-negotiable: original
dataset, all49 stocks, original InternVL3-2B, policy access to actual portfolio and
stock information, financial RL, and honest matched comparisons. Five-stock
results are diagnostic history only and cannot conclude the domain. The active
five-stock recovery worker was stopped, not left consuming training resources.

## Original dataset and immutable evidence

Source `research/data/nifty50_prices.csv`, SHA256
`29b847a33ad88b7568eef66e1fb9f3274a897bd3091ddb8551c129b75dd726f7`.
287263rows,49distinct tickers,1999-01-01 through2026-01-30. All49 have complete
observations throughout2018-2026Jan. Earlier unequal listing histories remain
preserved, with missing pre-listing cells rather than invented history. The
original file is unmodified. `original_panel.npz` retains every original price,
date and ticker without forward/backfill. No asset selected on returns.

Full audit: `research/full_universe_rl/original_dataset_audit.json`.
Do not quietly replace the primary period with February-November2026: those
dates are not in this original CSV. Any later extension must be separately
labeled supplementary data. Primary2025-Jan2026 preserves the original period.

## Supplementary execution records

Captured OHLC/dividend/split fields for all49 original tickers, retained separately
under execution_fields. LTIM's vendor lookup uses LTM following NSE's documented
symbol change, while the original identity remains LTIM:
https://nsearchives.nseindia.com/content/circulars/CML72948.pdf

Overlapping adjusted daily returns have no absolute differences above0.005
between the original CSV and supplementary adjusted-close capture. This is a
coarse consistency check, not proof of exact raw quote validity. Historical
prices can differ by adjustment factors; a vendor Close field is not automatically
the historical executable quote. Full-universe execution must address demergers
and fractional bonus issues. In particular Reliance2023 and ITC2025 require
separate entitlement accounting and quote verification. No silent stock exclusion,
unexplained adjustment credit, fictional executable quote or false profitability.

## Full-state model representation

One compact numerical state record per original stock, all49records before all49
action queries, plus ticker identities and account date/capital/cash/equity text.
Records include historical quote,held shares,position value,current weight,cash
affordability,trailing returns,volatility,drawdown,volume and account/cost fields.
All stock state records enter the unchanged original28-layer decoder through a
shared20->128->1536 input encoder; native vocabulary LoRA emits each action.
This additional input adapter is disclosed and is not an external action policy.
It must be saved/reloaded/ablated together with LoRA. No stock subset or truncation.

Action vocabulary is currently proposed0/0.5/1/2/4/7/10percent targets. This is
not the old five-stock action mapping and old checkpoints cannot be used under
it without adaptation. Need evaluate action resolution, small-budget feasibility,
caps,normalization and unused cash before declaring the design final.

Full text alone measured4086tokens; the compact state representation is being
tested on the actual GPU rather than assuming full coverage cannot fit. These
tests use artificial quotes/action labels and are mechanics-only, never financial
results. The numerical-only baseline and input-encoder ablation remain necessary
to distinguish language-model benefit from added numerical representation.

## Checks completed and pending

FiveCPUchecks pass: all49 observations/action coverage, future-data exclusion,
capital/holdings visibility,rejection of five-stock accounts, original-price
identity and canonical action mapping under reversed stock ordering. Every stock
state precedes every action query. GPU preflight is separate; inspect its receipt.

No full-universe training has been launched yet. Pending: verified historical
execution and corporate-action ledger, matchedholding baseline, full-state native
GPU forward/backward/reload, causal chronological train/validation/test contract,
learning gate, RL, baseline/ablation tests, actual-inference demo parity.
Do not call these pending checks complete or claim all errors eliminated.

The next task is resolving execution semantics while retaining the exact original
signal dataset and all49stocks. Never solve it by dropping difficult tickers.
User authorization is broad within that scope; no additional permission needed
for ordinary implementation, tests,training or reporting. Keep updating timeline
with measured numbers and distinguish plausible explanations from isolated causes.
