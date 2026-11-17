# Start here: final Finance RL artifact and demo

**Latest active work (17 November, afternoon):** state-aware run 2 finished and failed the primary test: trained −8.176%, untrained +1.893%, holding −1.795%. A causal supervised curriculum followed by own-account episodic RL is now running. Read [revision 3 research, rationale and worker handover](STATEFUL_RL_REVISION3_RESEARCH_20261117.md). No replacement is declared final yet; old results below are historical.

**17 November correction:** the old constant-mixture adapter does not meet the requested state-aware portfolio-management requirement. A new whole-share, full-stock/account-input RL attempt is running; read [the active-run handover](STATEFUL_RL_ACTIVE_RUN_20261117.md) first. The results below remain historical measurements, not a declaration that the requested domain is concluded.

**Selected artifact:** InternVL3-2B Nifty native static strategy-preference LoRA, seed 42, checkpoint 44. **Headline:** continuous retrospective test, 1 January 2025–30 January 2026: trained 13.28% return versus original 11.22% and equal holding 13.06%. Rebalanced equal weight at 13.44% remains stronger; holding superiority is uncertain and cost-sensitive. Training: 2019–2021; validation: 2022. No development score is the headline.

1. [Short final results](FINAL_RESULTS_SHORT.md).
2. [Detailed journal/method/failure audit](JOURNAL_RESEARCH_REPORT.md).
3. [Every measured policy/cost/period](NUMERIC_APPENDIX.md).
4. [Current paper-trading setup](../demo/PAPER_TRADING.md).
5. [Full historical context](CONTEXT_TRANSFER_FULL.md).
6. [Recovery archives and checkpoint backup status](../archives/README.md).
7. [Original SSD context bombs and college handover](ssd_historical_context/README.md).
8. [Fresh GitHub clone verification](RECOVERY_VERIFICATION.md).
9. [Completed checkpoint release and public-download verification](../archives/CHECKPOINT_RELEASE_STATUS.md).
10. [Native Finance tab in the existing Chat/Anomaly UI — 17 November update](FINANCE_UI_INTEGRATION_20261117.md).

The Finance menu now runs the selected Nifty adapter through actual model inference at every scheduled rebalance. It advances a historical market with simulated funds, not a broker/live feed. The whole 270-day path reproduces original research exactly. Finance conversation was removed after 60 actual responses from two model configurations failed the factual gate; the decision inspector is an explicit ledger, not model-generated reasoning. See explanation_assessment/.

Older reports sometimes name earlier artifacts final. They are retained history, superseded by this status and JOURNAL_RESEARCH_REPORT.md. Nothing establishes exhausted model potential, a flawless process, conventional-strategy superiority or journal acceptance. Other-domain trained weights/quality are not fully recertified.
