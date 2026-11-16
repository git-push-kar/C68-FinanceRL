# Finance RL · InternVL3-2B

A native RL adapter learns a mixture of causal stock-allocation strategies, with a reproducible historical paper-trading demo.

**Start:** [current context and results](docs/START_HERE.md) · [detailed research report](docs/JOURNAL_RESEARCH_REPORT.md) · [demo setup](demo/PAPER_TRADING.md) · [full context transfer](docs/CONTEXT_TRANSFER_FULL.md) · [recovery archives](archives/README.md).

## Selected result

Training2019–2021, validation2022, retrospective continuous test1January2025–30January2026. Same49stocks, costs and dates:

| Policy | Net return | Sharpe | Max drawdown |
|---|---:|---:|---:|
| Original model |11.22%|0.885|8.13%|
| Selected RL adapter |13.28%|1.033|7.78%|
| Equal-weight buy-and-hold |13.06%|1.012|7.75%|
| Equal-weight rebalanced |13.44%|1.035|7.79%|

Observed training benefit over raw; small, uncertain, cost-sensitive holding advantage. No established conventional-strategy superiority or untouched-test claim.

## Demonstration

`demo/` serves the portfolio menu at `http://127.0.0.1:8765/#finance`. Each scheduled rebalance calls the actual selected Nifty LoRA. Historical prices, simulated cash and fee-funded trades advance autonomously. The270-day model-backed account exactly reproduces the original measured path. Finance conversation was removed after factual testing; inspect actual execution records instead. Install Git LFS and the pinned original upstream base as described in [setup](demo/PAPER_TRADING.md).

![Actual paper-trading interface](demo/portfolio-paper-live.png)

## Repository organization

- `demo/`: current application, selected policy weights/data via LFS, tests and real inference/UI evidence.
- `docs/`: final results, journal-level method/failure report, complete numeric appendix, narration assessment and historical transfer.
- `research/`: source snapshots and canonical Nifty dataset for reproduction; historical scripts may retain original paths. Read recovery instructions rather than assuming every old script runs from repository root.
- `archives/`: historical source/metadata and selected packages via Git LFS, checksums, plus separate full-checkpoint release status.
- Original root-level scripts from `master` are retained legacy code and do not define the current selected native policy.

Use `git lfs pull --include="demo/adapters/finance_policy/**,demo/data/finance_paper/**"` for the minimal demo assets; plain GitHub ZIP downloads may contain LFS pointers rather than weights. All current commits use the authenticated user's configured identity; preexisting history is preserved.
