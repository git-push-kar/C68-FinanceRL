# Stateful supervised curriculum plus episodic Finance RL

Read [the complete design/research/limitations report](../../docs/STATEFUL_RL_REVISION3_RESEARCH_20261117.md).

This is the third state-aware attempt, not the old fixed-mixture adapter.
`train.py` runs causal supervised initialization and then group-relative clipped
policy-gradient updates on actual three-decision account episodes. It preserves
untrained/SFT/RL/holding comparisons; it never promotes a result automatically.

Run CPU tests before the GPU `--mode smoke` check. Both must pass before a fresh
`--mode train` run directory is created. Never overwrite a prior run. Use the
existing PolyvalentDemo Python environment and pinned local InternVL3 base.

Live run and logs are listed in the report. `status.json` supplies the actual
worker PID. The model sees complete state and emits native action-token logits.
The teacher supplies supervised labels only, with no inference fallback.

After completion, run `research/summarize_stateful_episode.py RUN_DIRECTORY` for
paired descriptive uncertainty intervals, exposure and capital/action comparisons.
Do not treat the adaptively reused retrospective tests as untouched confirmation.
