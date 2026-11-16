# Final Finance audit — 16 November 2026

This report supersedes checkpoint selections and template-narration claims in earlier handovers. No training is queued. The financial experiment is frozen. Unrestricted reliable explanations remain unresolved.

## Selected deployment

Run05 checkpoint176, adapter scale 1.0, full FP32 language inference. The full InternVL3-2B is retained, with vision on CPU during Finance inference and restored on adapter eviction. The original RL adapter produces the evaluated portfolio strategy preferences. A separate SFT adapter generates explanations; one checkpoint has not been demonstrated to do both jobs.

Runs01–03 continued the original attention/lm_head trading adapter and generated poor answers despite low synthetic loss. Run04 restarted from the pristine backbone using rank16 attention and MLP LoRA without lm_head: 18,464,768 trainable parameters, 768/96/96 synthetic train/validation/test records, 192 updates at learning rate 2e-4, selected validation response CE 0.006033792, approximately 1,279 seconds and 4.853 GiB peak training allocation. Run05 continued this branch with 384 retention and 384 harder examples, 192 updates at 8e-5, selected step176 CE 0.004000088, approximately 1,273 seconds and 4.888 GiB peak. Both used BF16 base training, FP32 adapters and gradient accumulation eight.

Architecture, initialization and data changed together; improvement cannot be attributed solely to removing lm_head. Synthetic accounts did not use final market rows as targets, but development failures informed task categories. Low template-distribution validation CE is not freeform factual accuracy.

## Selection evidence and limitations

The assistant's manual paired BF16 review of 30 actual-portfolio questions per model gave run04 15 pass / 5 partial / 10 fail and run05 15 pass / 4 partial / 11 fail. Both fully passed only 2/12 new-wording stress questions. Run05 was therefore not an overall BF16 winner. FP32 inference improved several accounting questions on a selected twelve-question difficult probe; scales 0, 0.5 and 0.75 were weaker on that probe. This supports a narrow deployment choice, not proof of the best possible configuration.

Twelve blank FP32 records in adapter_strength_review.json came from a CPU/GPU input-device bug and are invalid model-quality observations. The corrected probe is adapter_strength_fp32_review.json. Inputs now follow the language embedding device. Full-precision Finance inference uses approximately 6.7 GiB GPU memory. Other-domain full trained weights were not all available; unit residency checks do not certify all four domains.

## Post-selection actual HTTP audit

finance/evaluation/final_deployed_audit.json preserves six repeated regression questions and eight fresh post-selection phrasings, with drafts, revisions and checks. The eight fresh questions were not used to retune the model. This is a small developer audit, not independent universal accuracy measurement.

Successful regression answers explained the TLT buy and fees, UUP capital concentration versus estimated risk contribution, negative GLD/TLT P&L ordering, and allocation contribution versus investment return. After a model-generated revision, the benchmark answer correctly ranked standalone minimum variance above the trained model. Its statistical wording remained awkward. A holdings answer introduced unsupported wording about two trades.

Fresh-question successes included fee double-counting, weight drift, and the limits of counterfactual explanations. The answer about fixed strategy preferences was indirect. Important failures remained:

- Asked whether GLD's loss was larger than TLT's, it said “No” while quoting figures showing that it was larger. The checker missed this semantic reversal.
- Asked what performance could be attributed to fine-tuning, it gave an irrelevant transaction explanation with malformed “$1.93%” wording.
- Asked about a confidence interval containing zero, it invented a 0.01–0.03% preference interval. The checker flagged unsupported numbers, but the second model attempt remained wrong.
- The trade-versus-profit answer distinguished the concepts, but its specific transaction claims need record-level checking before being used as a showcase.

No failed answer was rewritten or removed. These results demonstrate actual local generation and several useful explanations, not reliable answers to any question. More of the same synthetic training is not supported by the present evidence. A future attempt needs a concrete generalization intervention and new independent evaluation.

## Runtime and verification

Start-FinanceDemo.ps1 launches http://127.0.0.1:8765/#finance using finance/active_model.json. The final HTTP audit took roughly 6–28 seconds per request after loading. Draft generation is buffered for limited checks and at most one model-generated revision before display. No checker or UI component substitutes preset answer prose. Residual detected problems trigger a visible warning; no detected problem does not imply correctness.

25 finance/residency/checker tests passed with one warning. Earlier API mock transport tests passed separately; these are software tests, not generated answer quality. All 413 original bundle hashes are unchanged. Evidence is preserved in finance/evaluation/integration_tests.log, original_bundle_verification.json and final_deployed_audit.json. The original financial bundle remains immutable.

## Frozen portfolio result

The RL policy learns fixed preferences over five external causal allocation strategies. It does not predict news or update its strategy probabilities every day. Lagged strategy calculations and prices change allocation targets.

The final 189-session February–October 2026 comparison is raw Sharpe 1.513051 versus trained three-seed mean 1.951463; total return about 6.532% versus 6.377%; maximum drawdown about 3.509% versus 2.490%. Standalone minimum variance has Sharpe 2.169516 and is stronger. The paired trained-minus-raw Sharpe 95% interval [-0.080094, 0.864912] includes zero. Observed risk-adjusted improvement is defensible; higher absolute return, statistically established superiority or superiority over classical optimization is not.

## Final presentation shortlist

1. Architecture: original Finance RL adapter, causal allocation/execution engine, and separate synthetic-SFT explanation adapter on the same fixed backbone. Explicitly disclose both Finance weight roles.
2. One financial comparison table: raw model, three-seed trained policy and standalone minimum variance, with return, Sharpe, drawdown, costs, dates and uncertainty. Include equity/drawdown curves. Do not imply the trained model had the highest return or beat the optimizer.
3. Historical portfolio UI: balance, holdings, activity, fees, date/seed controls and actual generated questions. No live brokerage claim.
4. Successful generated examples: TLT buy/fee correction, UUP concentration versus risk contribution, allocation contribution versus return. Preserve actual output and source dates.
5. Failure examples: fresh loss-comparison reversal, irrelevant attribution and fabricated confidence interval. Explain why synthetic validation loss and limited checking are inadequate reliability measures.
6. Full run history and paired comparisons belong in the appendix and handover. Final deployment selection is a narrow empirical choice, not universal optimality.

The portfolio experiment and real-model demo are presentable with these boundaries. The requested unrestricted justification reliability has not been achieved. Neither journal acceptance nor superiority over published systems is established by this audit.
