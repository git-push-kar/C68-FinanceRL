# Finance RL: verified candidate and training/validation/test results

## Recommendation

The Nifty native static strategy-preference adapter (seed42 checkpoint44) is the stronger candidate for demonstrating a training benefit. It improves test net return and Sharpe over the exact untrained policy, and slightly exceeds equal-weight buy-and-hold at the original10bp costs. Actual adapter weights were reloaded and verified. Do not claim it beats all simple trading strategies: rebalanced equal weight and inverse volatility remain stronger controls. No UI/deployment switch was made in this audit; the existing dashboard remains the old five-ETF experiment.

## Data partition for the final presentation

- Training:2019–2021, updates to LoRA parameters through financial RL.
- Validation:2022, checkpoint selection using net Sharpe. Selected steps42/123/456:44/40/44.
- Test:calendar2025 and January2026, shown separately. Also report a continuous1January2025–30January2026 test as supplementary material, carrying holdings through the year boundary.

2023–2024 development scores are EXCLUDED from the main results and figures. Their prior use for method design must remain disclosed in the methods/history. These test results are retrospective: the candidate was revisited after observing earlier experiments and the market data had been accessed elsewhere in the project. Changing terminology cannot turn them into independent untouched tests. Do not call validation or previously reused development history independent test evidence.

## Test2025:249 sessions,10bp fees

| Policy | Net return | Sharpe | Max drawdown |
|---|---:|---:|---:|
| Original untrained |13.274%|1.103|8.135%|
| Trained seed42 |15.074%|1.243|7.780%|
| Trained seed123 |15.074%|1.243|7.780%|
| Trained seed456 |15.074%|1.243|7.780%|
| Equal-weight buy-and-hold |14.864%|1.220|7.750%|
| Hold trained first target |14.741%|1.222|7.733%|
| Hold untrained first target |13.697%|1.136|8.120%|
| Equal-weight rebalanced |15.242%|1.244|7.787%|
| Inverse-volatility rebalanced |15.106%|1.252|7.754%|

The trained policy gains about1.80 percentage points versus untrained, and0.21 points versus equal-weight buy-and-hold. Its Sharpe gain over raw is about0.140. Its drawdown is slightly worse than buy-and-hold. All three trained seeds improve versus raw, but they share one market path and have near-identical strategies, so they are optimization replications, not independent market replications.

Conditional paired circular-block bootstrap,5000 resamples,20-session blocks: trained mean minus untrained return95% CI[0.590,3.264] percentage points; Sharpe CI[0.058,0.212]. Five/60-session sensitivity intervals also exclude zero for the raw comparison. These intervals do NOT adjust for historical method/candidate search and do not establish independent confirmatory significance. Versus equal-weight buy-and-hold, return CI[-0.880,1.277] points includes zero. Do not claim established superiority over holding.

## January2026 and continuous test

January2026 (21 sessions, independent cash reset): untrained-1.915%, trained-1.614%, buy-and-hold-1.588%. Training reduces the loss versus raw but does not beat holding. The month is too short for a strong standalone inference; bootstrap blocks comparable to or longer than this month are not credible independent confirmation.

Continuous1January2025–30January2026 (270 sessions, no year-end reset): untrained11.221%, trained13.283%, buy-and-hold13.059%, equal-weight rebalanced13.441%, inverse-volatility rebalanced13.343%. Report alongside the separate years; do not multiply reset-period returns and call them a continuous portfolio.

## Cost sensitivity

2025 test: at20bp, untrained12.563%, trained14.820%, buy-and-hold14.750%; at50bp, untrained10.458%, trained14.060%, buy-and-hold14.407%. The training gain over raw survives these modeled costs, but the small advantage over holding does not. These are fee sensitivity simulations, not a realistic market-impact model.

## What actually changed in the model

The same pinned full InternVL3-2B was loaded with FP32 active decoder and TF32 disabled; inactive original components remained on CPU. The six original packages (initial and selected for each seed) passed manifest verification. Maximum saved-versus-reloaded action-probability discrepancy was below1.5e-8. Disabling each trained adapter returned the same original untrained probabilities. This verifies real weight adaptation, rather than a manually substituted allocation.

For seed42, strategy probabilities changed approximately:
- Equal weight20.57% →32.88%.
- Inverse volatility31.39% →64.49%.
- Momentum30.48% →1.36%.
- Reversal17.56% →1.28%.

These probabilities are fixed strategy preferences; causal strategy target weights change with prices and the portfolio rebalances every five sessions. Training learned a more useful strategy mixture on these test histories. It did not learn daily news reasoning or prove state-dependent timing. The observed raw-relative gain includes lower turnover/costs, not exclusively superior gross stock selection.

## Verification and boundaries

Source stock-data SHA256 exactly matches the original training protocol. The original fixed49-stock universe and preprocessing remain; current-universe/survivorship and point-in-time limitations are not solved. All11 policies in each of nine period/cost combinations share exact return dates. Daily-return compounding and ending wealth reconcile; all hold baselines have zero turnover after the initial purchase. No training or original artifact modification occurred. Actual inference probabilities, full daily paths, bootstrap intervals, protocols and accounting checks accompany this report.

Main figures: presentation/nifty_test_2025.png, SVG and PDF; caption separate. Main presentation contains training/validation/test definitions and test results only. Historical development figures remain in prior archives for provenance.

## Defensible final statement

Financial RL changed the native2B model's strategy preferences and improved subsequent test return and Sharpe versus its original untrained policy across three training seeds. It slightly outperformed equal-weight buy-and-hold at the original fee setting over2025 and the continuous available test interval. That holding advantage is small, uncertain and cost-sensitive, and simple rebalanced strategies still match or outperform it. This supports an adaptation-benefit claim, not universal portfolio-management superiority.


---

# Historical research context: superseded primary ETF selection retained for provenance

# Current finalization update: read the closing addendum and final_presentation_figures/README.md. Earlier historical claims are superseded where inconsistent.

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


# Historical explanation development

# Finance: model-generated explanation specialization and handover

16 November 2026. This update supersedes the deterministic narration in portfolio
v2. The financial selection and final-test results from v2 are unchanged.

## What changed and why

The user explicitly required actual model-generated explanations, consistent with
the other Polyvalent domains. The old bounded reply engine did not meet that
requirement. It was removed from the deployed source, rather than relabeled as
reasoning. `finance/service.py` now only reads the verified replay evidence.

Inspection covered the existing InternVL AdapterService, residency manager,
Anomaly telemetry-to-language prompts, Deepfake forensic wrapper and API streaming
implementation. Finance now follows the shared model/LoRA lease mechanism. The
full pinned InternVL3-2B loads once, its finance explanation adapter attaches, and
its autoregressive decoder generates answer tokens from portfolio evidence and
conversation history. ChatML framing uses the model's own tokenizer template.

`finance/grounding.py` selects and formats relevant facts. It does not return an
answer, score a canned answer, retrieve teacher labels or look up previous replies.
`finance/generation.py` invokes real inference. Both streaming and nonstreaming
Finance endpoints require a real resident model and reject stub generation.
A failure produces an error, not a preset substitute. The frontend displays generated text after limited checks. Drafts and model revisions are included in the API receipt; the checker does not rewrite the prose.
Basic Markdown rendering escapes HTML before applying formatting.

## What the explanation means

The original portfolio policy learned fixed probabilities over five external
allocation strategies. Their price-based targets change with lagged inputs. The
model's policy prompt itself is constant. Therefore a faithful explanation can
identify the target, the signed transaction, strategy contributions, risk estimates
and scheduling rules; it cannot establish news reasoning or a fresh model forecast
that the original policy never produced.

Narration is a generated interpretation of records, not a recovered internal
thought transcript. The original RL adapter remains the source of the evaluated
portfolio mixture. A separate explanation LoRA is trained from the pristine backbone for language. The UI and model
configuration expose that distinction. No claim is made that the explanation copy
retains the original strategy probabilities or financial results if used as a new
trading policy. Demonstrating one checkpoint retaining both capabilities would
require an additional joint-training/policy-preservation experiment and evaluation.

## Failures investigated before further training

Actual generation was tested with the full pristine base and with the original RL
adapter. A verbose JSON evidence prompt caused repetitive output on a contrary
buy/sell question. More compact factual context improved fluency, but the original
weights still confused numerical quantities, called UUP a utility company,
misstated strategy probabilities and failed some optimizer comparisons.

The first synthetic SFT improved reference-number coverage, but this was not a
factual-accuracy score. Testing new questions showed remaining factual failures.
A second pass with focused evidence nearly memorized its narrow synthetic
validation answers: validation response loss fell to about0.033. It still produced
poor answers on genuinely different question types, including confusing account
returns with five-session returns and risk contributions with capital allocations.
This is why the second pass was not accepted on loss alone.

Decoding diagnostics compared risk context on/off with repetition penalty1.0.
Removing risk facts did not solve the problem and sometimes made errors worse.
Tokenization was checked: training ChatML token IDs and inference formatted-prompt
token IDs matched. The issue was not a different prompt tokenizer. Evidence labels
were clarified, particularly cumulative-versus-event fees, account period, Sharpe
units, and which pair the confidence interval compares.

## Training lineage and reproducibility

Pinned base: OpenGVLab/InternVL3-2B revision
899155015275a9b7338c7f4677e19c784e0e5a21. Base weight SHA256:
18b4cdbc8eab790f447feac717f4d0cecbb85caf8df198b3b2401c22506858e7.

Runs01-03 initialized from a copy of the frozen seed42 finance RL LoRA. Run04 uses a fresh explanation LoRA from the pristine backbone. The original
413-file bundle is never written by these scripts. Runs01-03 use 6,809,504 trainable LoRA parameters; run04 uses18,464,768. All use a BF16 frozen full backbone and FP32 trainable LoRA,
response-only cross entropy, batch1, gradient accumulation8, gradient clipping1,
AdamW weight decay0.01 and gradient checkpointing. Only response-position logits
are materialized for the loss, reducing memory without changing the response CE.

- Run01: 256 synthetic training examples,48 validation,48 test; two epochs,
  64 optimizer updates, peak learning rate6e-5. Selected update64 by validation CE.
- Run02: initialized from run01;384 training and48 validation examples; two epochs,
  96 updates, peak rate1e-4. Selected update80, not the worse final update96.
- Run03: initialized from run02;576 training,72 validation,72 test examples; two
  epochs,144 updates, peak rate6e-5. Selection uses minimum validation response CE.
  Final selection and actual-generation review are recorded below.

The third dataset varies strategy preferences instead of keeping the same mixture
in every teacher answer. It simulates correlated returns, calculates historical
volatility and shrunk covariance, solves bounded minimum variance and reconstructs
fee-funded trades. It adds explicit tasks about Sharpe versus return, novice
explanations, false news/action premises, rebalancing without a price forecast,
comparative holding P&L, drift, explanation attribution and variance contributions.

Synthetic records are instructional examples, not historical financial evidence
or complete simulated equity paths. Some summary fields are independently sampled;
they must not be used to evaluate investment performance. Training answers are
locally authored labels. Their existence does not make runtime generation a
lookup system: the serving code never reads the training examples or target text.

No actual final-period account rows or answers were used as SFT targets. However,
observed failures on real portfolio questions informed task categories and prompt
changes. Those repeated questions are development regression checks, NOT a blind
held-out language benchmark. This distinction must be kept in the paper.

Data manifests record split seeds/counts/hashes. Each run directory preserves the
protocol, source hash, progress, validation selection and all saved checkpoints.
The source used to format the third training set is snapshotted alongside it.
The working scripts are build_finance_synthetic*.py and train_finance_explanations*.py.

## Serving and user experience

Start-FinanceDemo.ps1 reads finance/active_model.json and sets actual base/adapter
paths with FINAL_STUB_GENERATION=0. Local Hugging Face loading is offline. The demo
venv reuses the installed finance ML dependencies through a Python3.12-compatible
site-packages link. The original research venv was not modified. First use loads
the model; subsequent questions reuse the resident base and adapter. Other tasks
can evict the finance adapter under the same existing lease lock.

The portfolio remains a historical replay. Customer-facing elements are value,
return, decline, fees, holdings and activity, with research evidence collapsed.
Questions and follow-ups are free text. Changing the portfolio date, seed or
starting balance resets history and cancels the old request. Sources show the
activity date, input cutoff, account date and frozen evidence identity. The model
receipt identifies actual generation and elapsed time. Financial actions are not
sent to a broker; there is no live order capability.

A separate 126-session historical risk context was added using the same lagged
window and covariance shrinkage as the allocator. This supports explanations of
concentration without pretending allocation weight equals variance contribution.
These additional derived fields do not alter any saved return or selected policy.

## Verification boundaries

The 20 finance/residency integration tests passed, and the original four API stub
transport tests passed separately. They check accounting, funding, causal context,
conversation transport, adapter routing, input validation and refusal to substitute
answers on errors. Mocked transport tests are explicitly test doubles; they are
not evidence of model answer quality. All413 original bundle hashes remain intact.

Actual-generation output files preserve mistakes as well as good replies:
- finance/evaluation/initial_generation_probe.json
- finance/evaluation/compact_context_probe.json
- finance/evaluation/validation_generation_comparison.json
- finance/evaluation/generation_v2_results.json
- finance/evaluation/decoding_diagnostic.json
- finance/evaluation/generation_v3_results.json (final pass)

Numeric-reference recall in these scripts only measures whether reference numbers
appear. It does not validate their relationships, sufficiency or factual correctness.
Manual review of the full generated text is necessary. The model can still err,
especially on unfamiliar multi-part questions, arithmetic and unsupported claims.
No exhaustive conversational reliability or other-domain weight certification is
implied. This checkout contains other-domain adapter configurations, but their full
trained weights are not all present; this update did not invent or replace them.

## Financial results remain unchanged

Final native RL test: 189 sessions in February-October2026, seed42 fixed before
final access, three trained seeds reported. Mean trained Sharpe1.951463 versus
initial1.513051; trained total return about6.377% versus initial6.532%; drawdown
about2.490% versus3.509%. Standalone minimum variance has Sharpe2.169516 and is
stronger. The paired95% interval for trained-minus-initial Sharpe includes zero.

The contribution is a demonstrated small-model RL strategy-preference adaptation
with improved observed risk-adjusted performance, not proof of superior active
management, statistical significance, news forecasting or a need for an LLM over
classical optimization. The explanation specialization is an additional interface
capability and must be reported separately from those investment results.


---

# Complete previous research context (historical v2)

The template-based narration statements below are superseded by the model-generation update above. Financial results remain unchanged.

# Polyvalent finance RL: context transfer and final presentation decision

Prepared 16 November 2026. This is the current finance handover. Older files called CURRENT_HANDOFF, FINAL_REPORT, STATUS or PENDING describe earlier stages; their titles do not establish current authority. Their original versions are retained under evidence. Read this report and the immutable native final bundle before launching anything.

## 1. Decision and its limits

Keep the five-strategy, constant-prompt native InternVL3-2B finance RL adapter, default strategy-token mapping, validation-selected update 64. Use seed 42 for the demonstration. Seeds 123 and 456 are replications, not alternate winners. The training changes the original model's strategy preferences and improves historical risk-adjusted performance versus that exact original policy. It is a completed, supportable finance-domain result, not evidence of a superior autonomous investment manager.

This choice is not supported only by a flattering delta over a poor baseline. Its absolute trained development Sharpe is approximately 1.5695, above the earlier four-strategy native, full-state native, head-only and numerical selector results on the same ETF period. The original policy already gives approximately 47.25% probability to minimum variance, so it was not reset to a weak artificial baseline. However, a reversed-label control scored approximately 1.5722 and the standalone optimizer approximately 1.7187. The accurate description is **the strongest supported primary native method among the completed comparable candidates**, not the highest number anywhere on the SSD.

The reversed mapping was a sensitivity control; choosing it now for a 0.0028 Sharpe difference would turn a robustness check into another selection opportunity. The classical optimizer is materially stronger and remains visible. Final outcomes are already consumed. This task did not retrain, change the reward, select a new seed, change the mapping or blend in more optimizer weight. It clarified the evidence and built a faithful portfolio interface around the frozen result.

## 2. Project intent and persistent requirements

Polyvalent is a shared-backbone, multiple-domain adapter project. The surrounding domains include ordinary conversation/logical reasoning, deepfake work and anomaly detection. Finance must remain the reinforcement-learning domain because the project has already been presented that way. Replacing finance RL with supervised financial question answering does not meet the user's requirement. InternVL3-2B and this desktop are fixed; datasets, action spaces, objectives and training approaches could change.

The user rejected explaining inferior results away with a smaller GPU. The continuation therefore preserved the full pristine model and addressed memory through execution placement rather than removing layers or silently changing active numerical precision. The objective became measurable adapter-attributable improvement with defensible comparisons, not a guarantee of publication or global investment superiority. There is no universal Sharpe threshold that makes a research result good: relevant matched controls, execution realism, uncertainty and the exact claim determine its evidential quality.

All earlier quiet-hour, fan-noise and idle-time restrictions were explicitly lifted. Do not reinstate historical scheduling guards. One-GPU-worker serialization, dependency waits and numerical checks still apply. No finance training is queued at this handover. The user permits independent jobs between check-ins; that is not a request for an unsolicited recurring automation.

## 3. Workspace map and evidence authority

| Location | Role |
|---|---|
| C:/Users/Parth/Documents/PolyvalentFinance | Authoritative local finance continuation: source, runs, logs and research reports |
| C:/Users/Parth/Documents/PolyvalentFinance/deliverables/finance_rl_native_final_20261116 | Immutable selected RL bundle: adapters, exact runtime, data, evaluation and receipts |
| C:/Users/Parth/Documents/PolyvalentDemo | Integrated demo checkout with the revised portfolio workspace |
| F:/Polyvalent_Continuation_2026-11-07 | SSD continuation backup, including the final native bundle |
| F:/START_HERE_POLYVALENT_HANDOVER.md | Original SSD entry point |
| F:/Polyvalent_Handover_2026-11-07 | Original context and transfer material |
| F:/Polyvalent_PC_Review_2026-11-07/ASSESSMENT.md | Initial local source/hardware assessment |
| F:/FinanceRL_College_2026-10-30.zip | College migration artifacts; distinguish these from source-only clones |
| F:/C68-FinanceRL and F:/FINANCE/C68-FinanceRL | Older finance lineages, not current adapter authority |
| F:/pc/anomaly | Newer local anomaly implementation and standalone Streamlit prototype |
| F:/pc/LY_C68 | Deepfake project and original pinned-base source; preserve local modifications |
| F:/pc/C68-LR | Logical-reasoning work; earlier audit found local commits ahead of remote |
| F:/pc/Polyvalent_Research_Paper_2026-10-31 | Earlier paper/evidence audit and remote source snapshots |

The integrated demo was found in git-push-kar/Polyvalent and cloned at master commit e3a48047104431ac853379e787b76942e257acd0. Finance was a placeholder. The local implementation adds a dedicated Finance iframe and API routes, preserving Chat and Anomaly. No remote push or publication was performed. The other domains' dependencies and model artifacts have not been jointly configured or benchmarked in this demo checkout. Tabs and adapters existing in source do not prove simultaneous residency, correct switching or cross-domain non-interference.

This report synthesizes continuation records, comparison JSON, protocol files, final artifacts and current code. It does not claim that every byte of every SSD backup was reread or every historical experiment rerun today. SOURCE_INDEX.json hashes 172 copied source documents/reports. ALL_COMPARISON_RUNS.csv extracts 138 per-run rows from reports with compatible initial/trained metrics. Legacy documents with different schemas remain in evidence rather than being treated as missing or zero results. The authoritative daily outputs and full checkpoints remain in their original run directories and final bundle.

## 4. Hardware and execution identity

Desktop: Ryzen 7600X, 16 GB RAM, RTX 4060 Ti 8 GB. Research Python is PolyvalentFinance/.venv/Scripts/python.exe. Base: local_models/InternVL3-2B, OpenGVLab revision 899155015275a9b7338c7f4677e19c784e0e5a21. Recorded base weight SHA-256: 18b4cdbc8eab790f447feac717f4d0cecbb85caf8df198b3b2401c22506858e7.

The active decoder uses FP32 on GPU with TF32 disabled. Inactive original components remain on CPU in BF16. No backbone layer removal, smaller replacement model or active-path quantization explains the result. Static native technical gates used about 5 GB GPU memory; full-state training approximately 5.1 GiB and around 350 seconds per update. These are branch-specific observations, not a universal speed guarantee. A constant prompt substantially reduces computation compared with repeated market-state prompting.

The final rank16 attention and original vocabulary-head LoRA has 6,809,504 trainable parameters. A separate head-only ablation actually trained 2,451,360; its initial gate counted temporarily re-enabled attention flags incorrectly. reports/readout_parameter_count_clarification.json corrects this without rewriting the original gate. Saved attention LoRA B matrices in that ablation remained zero. Do not copy the erroneous count into a paper.

Frozen-base checks, recovery of initial outputs with adapters disabled, exact fresh reload, actual shared finance runtime checks and packaged GPU inference support technical attribution. They do not establish market generalization or the necessity of a language model. Original environment/runtime receipts stay with the experiment; the demo has a separate lightweight environment with no Torch dependency.

## 5. Why earlier results looked contradictory

A Sharpe around 1.68 in an old presentation is not automatically a better version of the current method. Different assets, dates, horizons, heads, execution conventions and development reuse make absolute cross-experiment comparisons invalid. Some earlier covariance development rows exceed 2.0; that method was not the selected financial RL approach and failed its separate final test. Training fit, adaptively reused development performance and reserved evaluation answer different questions.

Early joint numerical-head/LoRA models also failed to isolate where the gain came from. Matched Nifty PPO improved by approximately +0.048879 on average, but its incremental difference against frozen-LoRA control was only about +0.002013, with a 20-session interval [-0.01185, 0.01712]. Whole-package improvement did not justify saying the adapter caused the gain. Some candidate strategies had path correlations around 0.967–0.999, leaving little distinct action signal. In other runs sampled-action gains disappeared under deterministic deployment execution.

These were attribution, objective, strategy diversity and execution-contract problems. The evidence does not isolate model size or GPU as their cause. The final method addresses several of them through native vocabulary probabilities, only LoRA trainable changes, an exact original-policy comparator, more distinct strategy tools, fee-funded execution and saved daily-output parity. It does not solve the separate challenge of useful market timing: the selected policy's learned preferences are constant.

## 6. Experiment chronology and failure interpretation

### 6.1 Original covariance campaign: preserve the failure

The risk-allocation campaign used 49 fixed stocks, 252-return covariance, numerical heads plus attention LoRA, 21-session rebalances and a 30% target cap. Development results looked strong, but many rows used zero signal delay. A legacy precision invariance audit failed, so strict FP32/TF32-off replications were retained. Final lagged execution, May2025–January2026, failed: trained mean Sharpe approximately -0.0816 and return -0.839%, against the initial policy around 1.2554 / +9.455%; equal weight approximately 1.2408 / +9.335%. The initial and equal-weight rows are different comparators.

Its reports and package are historical evidence. It did not satisfy the user's finance-RL requirement, and its final period must not be merged with the later native ETF final period. The historical CURRENT_HANDOFF copied into evidence describes that earlier failure. The source/cost/lag caveats attached to old development tables must travel with their headline values.

### 6.2 PPO, sector exposure and early native readout

Joint numerical-encoder/head/LoRA PPO gave ambiguous incremental LoRA benefit. Summary/numerical and sector-exposure branches explored sampled actions, soft actors, short-window direct objectives and full-history objectives. A sampled sector policy could improve while its deterministic deployment worsened. Soft actors and short-window variants did not provide repeatable improvement; full-history exposure64 had approximately -0.1605 mean Sharpe change. Chronological folds were mixed, not a consistent fix.

Native sector readout256 achieved only about +0.004125 with 1/3 positive seeds and broad uncertainty. These branches motivate better contracts and control attribution; they are not proof that all RL or all 2B financial models fail. The extracted CSV and original reports retain absolute values where available, including the less successful runs.

### 6.3 Nifty static native preference learning

Four original vocabulary choices represented causal strategies. Constant-prompt native training changed initial Sharpe 1.580558 to about 1.6983 on 2023–2024 development, mean gain +0.117706 across three seeds. Validation selected steps 44, 40 and 44. Return rose from about 44.110% to 45.617%, but drawdown worsened from about 11.932% to 12.416%. The 20-session interval included zero.

Historical chronology folds showed small positive gains (+0.0290 and +0.0341), again with intervals including zero. Overlapping training windows do not create independent replications or a stitched investable track record. A limited 90%-static/10%-regime branch was slightly better than its own initialization but slightly worse than the trained static anchor. Supervised warmup plus RL improved against raw, but RL itself worsened the warmup result by about -0.000982. This cannot be sold as a positive incremental RL result.

Rotated/reversed labels retained positive gains. Those were controls, not opportunities to choose a weak initial baseline. Nifty remains historical corroboration, not the primary ETF result or a directly rankable absolute score against it.

### 6.4 Four-strategy ETF studies

GLD/SPY/TLT/UUP/XLE was fixed before the development panel was fetched. The 2008–2024 panel contains 4,279 common sessions, without missing-price forward fill or outcome-selected asset dropping. Train2010–2018, validation2019–2021, adaptively reused development2022–2024.

Numerical static/dynamic selectors improved mean Sharpe about +0.129918/+0.159165, but timing-specific gains versus a validation-averaged fixed mixture were negligible or negative. Native static64 improved 0.899310 to 1.075087 mean, largely emphasizing inverse volatility. Full-state native16 improved 0.884158 to approximately 0.989069; timing gain over a fixed mix was only about 0.0016–0.0025. Head-only16 reached approximately 0.955464.

Those 16-update branches still improved on validation and are not claimed converged. A conditional extension to32 was prepared but not launched after the stronger five-strategy family became available. That is a bounded method/resource decision with a disclosed limitation, not proof that state-conditioned training cannot work.

### 6.5 Direct allocations, alternative rewards, anchors and annual refits

Direct-asset static/dynamic controllers deteriorated approximately -0.3478/-0.4167. Half-full-history/half-worst-training-year reward failed approximately -0.1407/-0.1261. A 90% causal minimum-variance anchor with learned adjustments underperformed its initial policy and pure optimizer, approximately -0.0609/-0.0515. Risk-and-cost reward alternatives were negative, approximately -0.0805/-0.1195. Economic plausibility of an objective did not guarantee useful optimization.

Annual refits used five training years and one validation year before each of 2022, 2023 and 2024. All18 CPU runs completed. Static yearly mean gains: -0.105246, 0, +0.148018; dynamic: -0.112007, 0, +0.240608. Both failed the declared native-promotion screen. No annual native campaign followed. Portfolios reset from cash each year and are not one continuous stitched strategy. An automatic launch block was subsequently authorized by the user and the work completed; no permission request remains pending.

### 6.6 Five-strategy risk selector: selected family

Adding a causal minimum-variance candidate to equal weight, inverse volatility, momentum and reversal improved static/dynamic numerical mean Sharpe by +0.409259/+0.520820. Dynamic timing itself added only approximately 0.0029–0.0053 over fixed-mixture comparison, below the declared0.01 screen; static native was promoted instead.

Primary native development Sharpe changed from1.242536 to1.569486 mean, three positive seeds, exact technical gates and a positive paired development interval. Label rotation/reversal also improved, without replacing the default mapping. This family was motivated by observed optimizer/control results and is an adaptive discovery, not an independently preregistered scientific hypothesis. The subsequent sealed test matters. The pure optimizer remains stronger on development and final data. The supported contribution is model adaptation and an auditable constrained strategy-selection interface, not a new optimizer or superior asset manager.

## 7. Absolute trained-quality audit

| Same ETF development history, 2022–2024 | Initial Sharpe | Trained Sharpe (mean) | Trained return | Trained max drawdown |
|---|---:|---:|---:|---:|
| multiasset_native_comparison64 | 0.899310 | 1.075087 | 25.316% | 7.551% |
| multiasset_dynamic_native_comparison16 | 0.884158 | 0.989069 | 25.453% | 8.153% |
| readout_only_ablation_comparison16 | 0.884158 | 0.955464 | 25.519% | 8.505% |
| risk_selector_static_comparison64 | 1.005529 | 1.414787 | 25.872% | 4.056% |
| risk_selector_dynamic_comparison64 | 1.005529 | 1.526349 | 25.992% | 3.471% |
| risk_selector_native_comparison64 | 1.242536 | 1.569486 | 25.937% | 3.250% |
| risk_label_rotate_comparison64 | 1.129322 | 1.566941 | 25.933% | 3.257% |
| risk_label_reverse_comparison64 | 1.064756 | 1.572239 | 25.937% | 3.238% |

All rows above share the ETF development period, but differ in action set, initialization, architecture or update budget. They answer whether the selected primary model is merely weak with a flattering delta: it is not. They do not isolate every causal factor in method design. The tiny reversed-label edge is disclosed, and neither that edge nor its larger delta justifies changing the final primary mapping. The pure minimum-variance development Sharpe is1.718731, higher than all these native rows.

Nifty's approximately1.698 and earlier covariance scores above2.0 are excluded from this ranking. The prospective final-choice record emphasized within-policy gain and technical checks; this supplementary absolute-quality audit clarifies the interpretation without rewriting the old choice record.

## 8. Exact selected method and action contract

The backbone scores existing A/B/C/D/E tokens under a literal constant prompt. Their normalized probabilities mix five external strategy target vectors. No teacher-action SFT warmup, financial action labels or new regression head train the selected model. Only attention and vocabulary-head LoRA change. The objective is full-history net-return Sharpe with0.01 KL regularization toward the initial policy; recorded attention learning rate1e-5, head3e-4, gradient clipping1 and KL guard. Three seeds train64 updates; validation every4, update0 eligible, earliest exact tie. All select64.

Describe this as direct financial reinforcement with a finite-path differentiable execution objective, not PPO or GRPO. The piecewise local derivative is not an unbiased population policy-gradient or global-optimality claim. A paper should explain the computation precisely rather than rely on the label RL alone.

Strategies: equal weight; half equal weight plus half126-return inverse volatility; half equal weight plus half126-session momentum skipping21; half equal weight plus half5-session reversal; and causal minimum variance. Minimum variance uses126 lagged returns, annualization252, 90% full covariance plus10% diagonal shrinkage, SLSQP from equal weights, asset bounds0.105–0.58, sum1, ftol1e-12, maximum200 iterations. Solver failures are rejected. Overall targets obey a0.60 cap; drifted holdings are not falsely guaranteed to retain that exact target cap between trades.

Seed42's learned mix is approximately1.16% equal weight,21.11% inverse volatility,1.05% momentum,1.02% reversal,75.67% minimum variance. Raw minimum-variance preference was already47.25%. Market data changes the strategy targets; it does not enter this policy's constant prompt. The shared environment's33-feature summary is not evidence of state conditioning in the final model.

Each decision combines target vectors, reconciles them with actual drifted holdings, funds fees internally and applies the next adjusted-price return. Signals end one close before execution. Rebalance every5 sessions; hold drifted positions otherwise. Start entirely in cash. Fees are10bp times the full stock-notional L1 of purchases plus sales, not half-turnover. Saved daily fields include net/gross return, fee, turnover, equity, drawdown and dates.

## 9. Reserved evaluation and interpretation

Return dates:2 February–30 October2026,189 common sessions. First execution30 January, signals through29 January. Method choice preceded final data acquisition; specification, seeds, dates, sources, artifacts, controls and criteria were sealed before evaluation. An exclusive access receipt was created before final portfolio outcomes. The panel contains711 dates with warmup. Original vendor outputs, adjusted CSV and hashes are retained. Completeness checks reject missing sessions rather than filling them.

| Policy | Net Sharpe | Total net return | Max drawdown |
|---|---:|---:|---:|
| Exact initial native policy | 1.513051 | 6.5318% | 3.5094% |
| Trained seed 42 | 1.951435 | 6.3770% | 2.4901% |
| Trained seed 123 | 1.952345 | 6.3772% | 2.4890% |
| Trained seed 456 | 1.950610 | 6.3769% | 2.4910% |
| Standalone minimum variance | 2.169516 | 6.3963% | 2.1951% |

Mean paired Sharpe gain +0.438412, three positive seeds. The20-session paired circular-block/seed-bootstrap95% interval is[-0.080094,0.864912], including zero. Bootstrap uses5,000 replicates;5/20/60-session diagnostics are retained. Shared time blocks preserve pairing; seed aggregation means average individual metrics, not an investable ensemble. Near-identical seeds show optimization repeatability, not three independent markets. Development's positive interval[0.034247,0.609301] is unadjusted for adaptive search and cannot replace final uncertainty.

The declared practical screen required meanΔSharpe≥0.05, at least2 positive seeds and no more than2 percentage points of mean return loss or extra drawdown. It passed. These are campaign thresholds, not universal financial standards. Total return fell approximately0.155 percentage points while maximum drawdown fell approximately1.019 points. This is risk-adjusted improvement, not higher profit. Seed42 turns simulated$100,000 into$106,377.03 with approximately$197.50 modeled fees.

Pure minimum variance's2.169516 final Sharpe must accompany the model result. Numerical static controls scored approximately1.7522,1.7529,1.7587, but their differing initialization prevents a matched claim of LLM necessity. Fees0/5/10/25bp preserve roughly+0.438 mean paired Sharpe gain. Ten paired sampled-action draws give9/10 positives per seed; those are secondary execution checks, not independent market replications or the primary expected-mixture execution.

Vendor adjusted prices are retrospective, not a point-in-time archive. Slippage, taxes, impact and liquidity are not fully modeled. Sharpe uses mean net daily return divided by population daily standard deviation times√252, with zero risk-free return. Fixed cash-rate sensitivities are illustrative, not observed excess-cash returns. Nine months is short. No live-profit guarantee, broad regime-generalization result or individual-stock-picking capability has been established.

## 10. Literature and contribution boundaries

[Moody and Saffell2001](https://pubmed.ncbi.nlm.nih.gov/18249919/) provides historical direct-financial-reinforcement context. This finite-path LoRA implementation is not a reproduction of their recurrent method. Minimum-variance optimization is conventional, not this project's new invention.

[FinATOM2026 preprint](https://arxiv.org/html/2608.09880v1) studies Llama3.2 1B token-based allocation on the same five ETF exposures, with causal mean–variance supervised initialization followed by policy optimization. It reports2023–2025 pooled net Sharpe1.394→1.494 under its5bp setup. That is SFT-to-policy, not pristine-base-to-adapter improvement. Dates, modalities, refitting, actions and execution differ; our1.95 cannot be called a victory over its1.494. The useful comparison is methodological feasibility of small-model financial optimization. Our stable tool mixture, direct path objective and exact adapter attribution are different experimental choices, not proof of novelty by themselves. The primary paper was reopened while preparing this handover.

A defensible framing is a modular2B adapter case study with matched initial policies, isolated trainable changes, failed-control disclosure, label sensitivity, deployment parity and a sealed final evaluation. Broader algorithmic novelty and LLM necessity remain unestablished. Cite influences; arbitrary changes made just to avoid similarity are not contributions. Four domains/adapters alone do not settle originality.

## 11. Portfolio product and explanation design

Finance now opens on portfolio value, net gain, largest decline from peak and total fees. A chart shows value through the selected date, with optional untrained and minimum-variance comparisons. Holdings show current drifted exposures, values, weights and cumulative market P&L before fees. Activity lists buys/sells, one fee per rebalance, asset filters and a link to explain that activity. Holdings/activity export toCSV. Starting capital scales the same simulated path; it does not retrain or produce a new experiment.

All-run research tables, strategy-contribution heatmaps and seed controls are removed from the primary customer flow. A collapsed evidence panel keeps limitations, full-period comparisons and replication selection. The historical-paper badge stays visible. No fictitious broker connection, live quote or order capability is implied. The outer Chat/Anomaly tabs remain. No-adapter-loaded can be correct: replay does not require live GPU inference.

The exporter checks every frozen-bundle hash, reconstructs actions and verifies daily-return parity, then independently reconstructs all daily holdings and market P&L from the same execution kernel. For every3×189 daily observation, assets+cash equal equity and initial capital+cumulative market P&L−fees equal equity. All target contributions sum correctly; trades fund fees and purchases from the recorded account. Floating-point tolerances are applied, not fake residual profits.

The as-of view truncates charts, holdings, fees, risk and available activity consistently. A rebalance fee enters the original ledger in its first post-execution daily return; activity appears only once its daily_start observation is included. This avoids displaying a trade before its fee is reflected in balance. The separate research panel explicitly labels full-period results. As-of handling is explanation correctness, not security: the replay endpoint serves the entire historical research simulation.

The assistant resolves tickers and common names: gold, Treasuries/bonds, energy, dollar and US equities. It can answer multiple requested facts, remember the last unambiguous asset for follow-ups, use a recordedYYYY-MM-DD execution date, correct contrary buy/sell premises and reject future/unavailable activity. Answers cite activity, signal cutoff, as-of date and frozen seal. Supported evidence includes signed trades, prior/target weights, dominant strategy contributions, costs, current holdings, cumulative market P&L, risk and initial-policy comparisons.

Example:9 March2026 seed42 bought$560.74 TLT per$100,000 starting capital. Weight changed20.90%→21.44%; minimum variance contributed16.07 percentage points and inverse volatility4.74. Signals stop6 March, fee$2.25. A question saying it sold bonds is corrected. At10 March close, TLT drifted to approximately$21,886.12 /21.27%; a follow-up about current worth uses that daily value, not the old target. Outcomes after a trade are retrospective, limited to the displayed date, and never presented as information available at decision time.

### Explanation capability and further specialization

This is a deterministic evidence-grounded assistant, not unrestricted generative reasoning from the finance adapter. It explains the actual mechanism linking strategy outputs to trades. Contribution arithmetic is not feature importance, private model reasoning or the causal effect of a trade versus another action. It cannot supply news-based reasons, unknown-stock decisions, live forecasts, tax-lot accounting or broker orders. Unfamiliar phrasing can require clarification; this is defined domain coverage, not universal natural-language understanding.

No additional finance training is required for these rule/trade/accounting explanations. The finance RL adapter was not trained to generate rationales. If broad conversational fluency or complex counterfactuals are later required, add a separate narration/tool layer with structured evidence, numerical/date verification, abstention and adversarial evaluation. A general/reasoning adapter could narrate, but that capability must not be attributed to the finance RL objective. Rationale fine-tuning without evidence checks can increase fluent fabrication. Genuine counterfactual analysis needs an explicit alternative action, the same execution kernel and clear hypothetical labeling; it is not implemented here.

## 12. Verification and reproducibility

Original research checks include frozen base, disabled-adapter recovery, selected-checkpoint reload, actual finance runtime, packaged GPU prediction and six original/trained CPU replay paths. The immutable bundle manifest covers413 files. This task rebuilt replay from it and added daily accounting verification; it did not rerun training or replace frozen outcomes.

The demo unit suite checks funding/contributions, every daily balance/P&L identity, API validation, all seed42 trade-direction premise corrections, aliases, follow-ups, combined fee/trade questions, no-future as-of outcomes, unsupported assets/orders, explicit dates and capital scaling. Browser verification checks actual visible balances, dated explanations and follow-up values. See VERIFICATION.md for the final test and browser record. These checks do not claim complete language coverage, production security or other-domain functionality.

From C:/Users/Parth/Documents/PolyvalentDemo:

```powershell
& .\.venv\Scripts\python.exe -m uvicorn api.server:app --host 127.0.0.1 --port 8765
```

Open http://127.0.0.1:8765/#finance. The server is localhost-only. After editing replay or service restart it because evidence is cached. Use the demo environment for FastAPI and the finance environment for reconstruction/Torch:

```powershell
& C:\Users\Parth\Documents\PolyvalentFinance\.venv\Scripts\python.exe finance\build_replay.py --bundle C:\Users\Parth\Documents\PolyvalentFinance\deliverables\finance_rl_native_final_20261116
& .\.venv\Scripts\python.exe -m unittest tests.test_finance_workspace -v
```

For actual GPU prediction follow the immutable bundle README/predict.py. Set FINANCERL_EXECUTION=selective_fp32; FINANCERL_BASE_MODEL to the pinned local base; HF_HUB_OFFLINE=1; TRANSFORMERS_OFFLINE=1; OMP_NUM_THREADS=2; MKL_NUM_THREADS=2. Do not substitute a merged base. The runtime outputs weights, never broker orders. Exact CPU replay of an extracted constant policy is valid when labeled honestly.

Critical files under the finance root:

- NATIVE_FINAL_METHOD_CHOICE.json: prospective primary family, seeds, criteria, controls; do not rewrite after outcomes.
- NATIVE_FINAL_EVALUATION_SPEC.json and checks/NATIVE_FINAL_20261116/contract.json: sealed dates, protocol, source and artifact identities.
- reserved_finance_native_2026_access.json: final-period consumption receipt.
- checks/NATIVE_FINAL_20261116/result.json, seed42/123/456.json, controls.json, secondary_sensitivity.json: authoritative final outcomes.
- deliverables/finance_rl_native_final_20261116/adapters/seed42: fixed demonstration adapter; other seeds are replications.
- native_finance_runtime.py: causal controller and hold/rebalance contract.
- reports/risk_selector_native_comparison64.json: primary development comparisons and selection steps.
- reports/FINAL_CAMPAIGN_STATUS.json: late-stage campaign ledger, not the entire earliest history.
- RL_CONTINUATION.md: chronological decisions; newer completion entries supersede old pending text.

Demo files: finance/build_replay.py, finance/service.py, api/finance.py, ui/finance.html, tests/test_finance_workspace.py and FINANCE_WORKSPACE.md. The immutable research package, updated demo archive and context-transfer package have different purposes. Do not change the research bundle to update a UI. Include ignored replay data explicitly in archives; do not bundle caches, virtual environments or a substitute base.

## 13. Completed late-stage campaign ledger

All rows, including negative and zero outcomes, are retained. Periods differ: no global ranking. ALL_COMPARISON_RUNS.csv adds per-run absolute values and checkpoint choices from compatible reports, including earlier branches. SourceJSON contains full paths and additional intervals.

| Method | Period | Mean ΔSharpe | Positive seeds | Paired 95% interval, 20 sessions | Source |
|---|---|---:|---:|---|---|
| Nifty static native | 2023-01-02–2024-12-31 | +0.117706 | 3/3 | [-0.064579, 0.337915] | static_native_seed_comparison_budget64.json |
| Nifty limited timing | 2023-01-02–2024-12-31 | +0.009797 | 3/3 | [-0.007620, 0.031525] | regime_native_seed_comparison_budget64.json |
| Nifty warmup then RL (vs warmup) | 2023-01-02–2024-12-31 | -0.000982 | 0/3 | [-0.003749, 0.001703] | warmup_native_seed_comparison_budget64.json |
| Five-asset static numerical | 2022-01-03–2024-12-31 | +0.129918 | 3/3 | [-0.054670, 0.313552] | multiasset_static_comparison64.json |
| Five-asset dynamic numerical | 2022-01-03–2024-12-31 | +0.159165 | 3/3 | [-0.055058, 0.374260] | multiasset_dynamic_comparison64.json |
| Five-asset static native | 2022-01-03–2024-12-31 | +0.175776 | 3/3 | [-0.049332, 0.397205] | multiasset_native_comparison64.json |
| Five-asset dynamic native | 2022-01-03–2024-12-31 | +0.104912 | 3/3 | [-0.052643, 0.263644] | multiasset_dynamic_native_comparison16.json |
| Direct asset weights, static CPU | 2022-01-03–2024-12-31 | -0.347751 | 0/3 | [-0.787572, 0.098033] | asset_allocation_static_comparison64.json |
| Direct asset weights, dynamic CPU | 2022-01-03–2024-12-31 | -0.416709 | 0/3 | [-0.894917, 0.062370] | asset_allocation_dynamic_comparison64.json |
| Worst-year reward, static CPU | 2022-01-03–2024-12-31 | -0.140716 | 0/3 | [-0.508611, 0.239958] | asset_robust_static_comparison64.json |
| Worst-year reward, dynamic CPU | 2022-01-03–2024-12-31 | -0.126146 | 0/3 | [-0.482496, 0.242488] | asset_robust_dynamic_comparison64.json |
| Causal risk anchor, static CPU | 2022-01-03–2024-12-31 | -0.060907 | 0/3 | [-0.160886, 0.042580] | risk_anchor_static_comparison64.json |
| Causal risk anchor, dynamic CPU | 2022-01-03–2024-12-31 | -0.051511 | 0/3 | [-0.155944, 0.057534] | risk_anchor_dynamic_comparison64.json |
| Nifty rotated labels native | 2023-01-02–2024-12-31 | +0.131845 | 3/3 | [-0.078389, 0.389700] | label_rotate_comparison64.json |
| Nifty reversed labels native | 2023-01-02–2024-12-31 | +0.157864 | 3/3 | [-0.102689, 0.473880] | label_reverse_comparison64.json |
| Five-asset readout-only native | 2022-01-03–2024-12-31 | +0.071306 | 3/3 | [-0.039188, 0.184329] | readout_only_ablation_comparison16.json |
| Risk-cost reward, static CPU | 2022-01-03–2024-12-31 | -0.080508 | 0/3 | [-0.371579, 0.204371] | asset_risk_reward_static_comparison64.json |
| Risk-cost reward, dynamic CPU | 2022-01-03–2024-12-31 | -0.119527 | 0/3 | [-0.411189, 0.173225] | asset_risk_reward_dynamic_comparison64.json |
| Risk-tool selector, static CPU | 2022-01-03–2024-12-31 | +0.409259 | 3/3 | [0.119728, 0.706132] | risk_selector_static_comparison64.json |
| Risk-tool selector, dynamic CPU | 2022-01-03–2024-12-31 | +0.520820 | 3/3 | [0.123802, 0.905237] | risk_selector_dynamic_comparison64.json |
| Risk-tool selector, native | 2022-01-03–2024-12-31 | +0.326950 | 3/3 | [0.034247, 0.609301] | risk_selector_native_comparison64.json |
| Annual static CPU 2022 | 2022-01-03–2022-12-30 | -0.105246 | 0/3 | [-0.419934, 0.232557] | multiasset_annual_static_2022_comparison64.json |
| Annual static CPU 2023 | 2023-01-03–2023-12-29 | +0.000000 | 0/3 | [0.000000, 0.000000] | multiasset_annual_static_2023_comparison64.json |
| Annual static CPU 2024 | 2024-01-02–2024-12-31 | +0.148018 | 3/3 | [-0.021278, 0.332799] | multiasset_annual_static_2024_comparison64.json |
| Annual dynamic CPU 2022 | 2022-01-03–2022-12-30 | -0.112007 | 0/3 | [-0.461955, 0.257701] | multiasset_annual_dynamic_2022_comparison64.json |
| Annual dynamic CPU 2023 | 2023-01-03–2023-12-29 | +0.000000 | 0/3 | [0.000000, 0.000000] | multiasset_annual_dynamic_2023_comparison64.json |
| Annual dynamic CPU 2024 | 2024-01-02–2024-12-31 | +0.240608 | 3/3 | [0.035615, 0.476470] | multiasset_annual_dynamic_2024_comparison64.json |
| Risk selector rotated-label control | 2022-01-03–2024-12-31 | +0.437619 | 3/3 | [0.068937, 0.794249] | risk_label_rotate_comparison64.json |
| Risk selector reversed-label control | 2022-01-03–2024-12-31 | +0.507483 | 3/3 | [0.098467, 0.906643] | risk_label_reverse_comparison64.json |

## 14. Earlier development table, preserved for context

The following is the original covariance/numerical campaign table. Its apparently higher scores must not disappear from the handover, but its own lag/protocol caveats matter. It is not the selected native ETF final result.

# Recorded campaign results

2026-11-07T16:48:05.555131+05:30

reused 2023–2024 development; lag1-trained utility controls use one-close delay, other rows retain zero lag; no final-period results

| Run | Step | Signal delay | Return | Sharpe proxy | Max drawdown | ΔSharpe vs initialization |
|---|---:|---:|---:|---:|---:|---:|
| covariance_lora_mlp_seed42 | 512 | 0 | 50.928% | 2.0783 | 14.222% | +0.4273 |
| covariance_lora_only_seed42 | 384 | 0 | 50.733% | 2.0953 | 13.167% | +0.6392 |
| covariance_no_lora_seed123 | 512 | 0 | 50.422% | 2.0898 | 12.292% | +0.4272 |
| covariance_no_lora_seed42 | 512 | 0 | 50.293% | 2.0757 | 13.329% | +0.4172 |
| covariance_no_lora_seed456 | 384 | 0 | 52.827% | 2.1136 | 13.859% | +0.4527 |
| covariance_risk_no_lora_seed123 | 384 | 0 | 51.667% | 2.1688 | 12.592% | +0.5062 |
| covariance_risk_no_lora_seed42 | 384 | 0 | 49.478% | 2.0902 | 13.291% | +0.4317 |
| covariance_risk_no_lora_seed456 | 384 | 0 | 48.875% | 2.0638 | 13.072% | +0.4029 |
| covariance_risk_seed123 | 512 | 0 | 53.899% | 2.2618 | 12.467% | +0.5992 |
| covariance_risk_seed42 | 384 | 0 | 51.317% | 2.1480 | 13.513% | +0.4895 |
| covariance_risk_seed456 | 512 | 0 | 48.055% | 2.0434 | 12.573% | +0.3825 |
| covariance_risk_strict_no_lora_seed123 | 384 | 0 | 51.710% | 2.1704 | 12.584% | +0.5079 |
| covariance_risk_strict_no_lora_seed42 | 384 | 0 | 48.802% | 2.0607 | 13.330% | +0.4023 |
| covariance_risk_strict_no_lora_seed456 | 384 | 0 | 48.849% | 2.0631 | 13.082% | +0.4021 |
| covariance_risk_strict_seed123 | 512 | 0 | 53.856% | 2.2599 | 12.477% | +0.5973 |
| covariance_risk_strict_seed42 | 384 | 0 | 50.211% | 2.1167 | 13.523% | +0.4582 |
| covariance_risk_strict_seed456 | 512 | 0 | 47.898% | 2.0366 | 12.585% | +0.3756 |
| covariance_teacher_seed123 | 512 | 0 | 51.013% | 2.1348 | 12.368% | +0.4723 |
| covariance_teacher_seed42 | 384 | 0 | 51.454% | 2.1249 | 13.248% | +0.4664 |
| covariance_teacher_seed456 | 384 | 0 | 52.139% | 2.1310 | 13.519% | +0.4701 |
| numeric_attention_seed123_lr0003 | 128 | 0 | 47.766% | 2.0549 | 13.447% | +0.3945 |
| numeric_attention_seed42_lr0003 | 512 | 0 | 47.798% | 1.9870 | 13.630% | +0.3258 |
| numeric_attention_seed42_lr001 | 256 | 0 | 48.649% | 2.0165 | 13.778% | +0.3553 |
| numeric_attention_seed456_lr0003 | 512 | 0 | 54.832% | 2.1930 | 13.593% | +0.5319 |
| numeric_covariance_seed123 | 256 | 0 | 50.820% | 2.1323 | 13.203% | +0.4715 |
| numeric_covariance_seed42 | 384 | 0 | 45.112% | 1.9382 | 13.527% | +0.2761 |
| numeric_covariance_seed456 | 512 | 0 | 48.375% | 2.0553 | 13.378% | +0.3951 |
| numeric_risk_seed123 | 384 | 0 | 49.053% | 2.0624 | 13.525% | +0.4020 |
| numeric_risk_seed42 | 512 | 0 | 52.429% | 2.1676 | 13.097% | +0.5064 |
| numeric_risk_seed456 | 512 | 0 | 49.495% | 2.0574 | 13.716% | +0.3963 |
| numeric_utility_lag1_seed123 | 32 | 1 | 46.830% | 2.0365 | 13.300% | -0.0594 |
| numeric_utility_lag1_seed42 | 64 | 1 | 43.915% | 1.8710 | 14.097% | -0.1389 |
| numeric_utility_lag1_seed456 | 128 | 1 | 51.431% | 2.1060 | 12.941% | -0.1106 |
| numeric_utility_seed123 | 96 | 0 | 44.958% | 1.9353 | 13.523% | -0.1197 |
| numeric_utility_seed42 | 64 | 0 | 43.366% | 1.8469 | 14.150% | -0.1401 |
| numeric_utility_seed456 | 128 | 0 | 50.687% | 2.0686 | 13.045% | -0.1243 |
| utility_seed42 | 32 | 0 | 50.122% | 2.0969 | 12.830% | -0.0280 |

Selection criteria and all checkpoints remain in each run folder.
Read LITERATURE_COMPARISON.md for published-method scope and classical controls.
This table alone does not establish statistical superiority or equivalence.


## 15. Remaining work and restart boundaries

Finance training and sealed final evaluation are concluded for this artifact. Remaining useful work is paper synthesis, other-domain integration and cross-domain evaluation. A future scientific extension needs new independent evidence or explicitly exploratory reanalysis; consumed data cannot become unseen again. Do not relaunch old drivers because a document title says PENDING.

The customer-oriented interface is a historical portfolio demonstration, not a complete commercial platform. Brokerage authentication, real deposits/withdrawals, tax lots, suitability profiles, live execution, data monitoring and operational controls are not implemented. Do not add invented live transactions to make a demo seem autonomous. Shared GPU switching and cross-domain non-interference for all four actual adapters remain separately unverified in this checkout.

Explanations cover the recorded process. No rationale-specialized model was trained. A presenter asked about a market news event should state that news is not an input. Asked why not simply use minimum variance, state that it was stronger here; this work demonstrates reward-trained modular adaptation, not an economic necessity for a language model. That limitation should remain in the paper and not merely in this handover.

## 16. Final boiled-down presentation set

1. **Architecture slide:** pristine InternVL3-2B → rank16 attention/readout LoRA → existing A–E token probabilities → five causal strategy targets → exact fee-funded portfolio ledger. Label constant preferences and direct financial reinforcement. Do not draw news/state inputs into the constant prompt or call it PPO/GRPO.
2. **Protocol slide:** GLD/SPY/TLT/UUP/XLE; train2010–18, validation2019–21, reused development2022–24, reserved Feb–Oct2026; three seeds; update0 eligible;64 selected updates; one-close signal delay; five-session rebalances;10bp full-L1 fees. Full2B retained;6,809,504 trainable parameters.
3. **Primary results table:** development raw1.242536→trained mean1.569486; final raw1.513051→trained mean1.951463. In the same final table show returns6.5318%/6.3770%, drawdowns3.5094%/2.4901%, optimizer Sharpe2.169516 and all three seeds or their range. Include meanΔSharpe+0.438412 and95% interval[-0.080094,0.864912]. Do not omit lower return, optimizer or uncertainty.
4. **Final curve figure:** immutable final_performance.png/svg with raw, trained and classical control, dates and cost convention. Historical net simulations, not live returns. Seed42 is pre-fixed, not the best final seed.
5. **Small ablation/robustness table:** four-strategy native trained Sharpe~1.0751 versus five-strategy primary~1.5695 on the same development period; positive gains under alternate labels; frozen-base/disabled-adapter/reload checks. More numerical, head-only, timing and fee controls belong in supplement. Strategy-set differences do not isolate novel optimization or LLM advantage.
6. **Portfolio demo:** final-date$100,000 account→$106,377.03; holdings, fees and activity. Switch to10 March, ask why bonds were sold, show the corrected buy and signal dates. Follow with “How much is it worth?” and show that current values agree. Identify verified historical replay, not live GPU inference during the UI session.
7. **Balanced conclusion:** “Financial-reward training of an isolated InternVL3-2B adapter improved risk-adjusted portfolio performance over its exact initial policy across three seeds. Reserved-period point estimates remained positive with lower drawdown and slightly lower return; uncertainty includes zero and standalone minimum variance remained stronger.”

**What makes this batch presentable:** native adapter-only attribution, an unmodified raw baseline, meaningful absolute trained improvement over comparable primary native branches, three-seed repeatability, label controls, exact reload/deployment checks, a prospectively sealed final evaluation, and a portfolio interface whose explanations reconcile with actual execution. A single winning Sharpe is not the argument.

**Exclude from the main presentation:** every failed trial, every sampled rollout, annual-refit year grids, old covariance peaks, Nifty slides as a direct ETF comparison, infrastructure logs and full hashes. Summarize the search and make this report/appendix available. Negative controls explaining selection should remain visible; individual hyperparameter archaeology need not occupy the main PPT or paper.

**Avoid claims of:** state-of-the-art trading, outperforming FinATOM, higher profit after training, statistically confirmed final superiority, autonomous market reasoning, proven LLM necessity, convergence of abandoned16-update branches or fully verified four-domain production deployment. None is established by these results.


---

## Final presentation shortlist

1. Architecture: original Finance RL adapter, causal allocation/execution engine, and separate synthetic-SFT explanation adapter on the same fixed backbone. Explicitly disclose both Finance weight roles.
2. One financial comparison table: raw model, three-seed trained policy and standalone minimum variance, with return, Sharpe, drawdown, costs, dates and uncertainty. Include equity/drawdown curves. Do not imply the trained model had the highest return or beat the optimizer.
3. Historical portfolio UI: balance, holdings, activity, fees, date/seed controls and actual generated questions. No live brokerage claim.
4. Successful generated examples: TLT buy/fee correction, UUP concentration versus risk contribution, allocation contribution versus return. Preserve actual output and source dates.
5. Failure examples: fresh loss-comparison reversal, irrelevant attribution and fabricated confidence interval. Explain why synthetic validation loss and limited checking are inadequate reliability measures.
6. Full run history and paired comparisons belong in the appendix and handover. Final deployment selection is a narrow empirical choice, not universal optimality.

The portfolio experiment and real-model demo are presentable with these boundaries. The requested unrestricted justification reliability has not been achieved. Neither journal acceptance nor superiority over published systems is established by this audit.


---

# Finalization and full-period figure addendum — 16 November 2026

## Decision

Freeze the selected original Finance RL adapter as the final artifact for this campaign, and report a mixed outcome. This is a decision to conclude the current experiment, not proof of optimality or consistent financial superiority. Do not describe 6.38% as a model-wide return ceiling or the highest-return run. It is the selected adapter's return on the original reserved 2026 interval. Explanation reliability remains a separate unresolved limitation; finalization does not mean every original quality objective was achieved.

## What data belongs in the results

RL weight training used 2010–2018. Validation 2019–2021 selected checkpoints. Development 2022–2024 was reused adaptively to design/select methods. Thus “not used for gradient training” does not mean untouched or independent. Show validation/development only with those labels. The main generalization result remains the original reserved 2 February–30 October 2026 test, chosen before evaluation. Report three trained seeds; deployment seed42 was fixed beforehand, not chosen from test scores.

Include all completed comparable frozen-adapter additional annual evaluations (2022, 2023, 2024, partial 2025) alongside that main result, including negative outcomes. The 2025 data had been accessed in other project campaigns and is an additional retrospective audit, not a new pristine test. Its available saved panel lacks sufficient warmup for January1 under the unchanged environment, so evaluation starts January10. Do not silently treat it as a full calendar year.

Annual portfolios reset to cash. They cannot be multiplied or stitched into one continuous investable result; such a result would require a separate continuous simulation with holdings carried across boundaries. Do not pool development and test into a headline score. Do not select another checkpoint or dates because they produce a prettier chart. No new evaluation of validation years or every available historical date is claimed by this figure package. All completed same-policy comparisons are included in its CSVs; historical experiments on other assets/methods remain in the main run history, not falsely merged with this adapter.

## Figures for PPT/reports

01_final_metrics: six panels for net return, zero-risk-free annualized Sharpe, maximum drawdown, annualized volatility, average daily full-L1 turnover and modeled dollar fees. Identical 189-session dates and execution for untrained, trained seed42, standalone minimum variance and equal weight. Mean turnover includes non-rebalance days. Fees are already deducted from returns. Volatility is population daily standard deviation times sqrt(252). The three-seed mean Sharpe and paired interval are in the caption; the bars use deployment seed42 consistently. Full metrics for all seeds are supplied separately.

02_final_equity_drawdown: actual saved daily portfolio values and drawdowns, beginning with $100,000 cash before the first return. No smoothing or synthetic path.

03_period_robustness: all five completed yearly/partial-year periods, separately labeled development, additional and reserved. Return is period total, not annualized. All annual comparisons use the same frozen seed42 policy against its exact untrained probabilities. Dates, period lengths, simple baselines and remaining metrics are in period_metrics.csv and the original annual audit.

Each figure is supplied as high-resolution PNG, vector SVG and PDF. CSVs contain the numbers; SOURCE_MANIFEST.json identifies input-file SHA256 hashes. make_final_finance_figures.py reproduces figures. No base or adapter weight was changed.

## Final interpretation and numbers

Period: untrained return → trained return; untrained Sharpe → trained Sharpe; untrained max drawdown → trained max drawdown.
- 2022 development: 4.585% → 4.965%; 0.583 → 0.807; 5.047% → 3.250%.
- 2023 development: 7.445% → 7.004%; 1.380 → 1.597; 3.739% → 2.982%.
- 2024 development: 12.225% → 12.057%; 2.325 → 2.810; 3.170% → 2.481%.
- Partial 2025 additional audit: 11.914% → 8.560%; 1.540 → 1.273; 6.258% → 5.792%.
- Original 2026 reserved test, seed42: 6.532% → 6.377%; 1.513 → 1.951; 3.509% → 2.490%.

Trained three-seed mean final Sharpe1.951463, raw1.513051. Paired95% ΔSharpe interval [-0.080094,0.864912] includes zero. Standalone minimum variance final Sharpe2.169516, return6.396%, drawdown2.195%; it outperforms the learned portfolio on all three of these metrics in this interval. The selected trained policy is not superior to this simple comparator.

Across the five reported periods, trained drawdown was lower in5/5; net return higher in1/5; Sharpe higher in4/5. These are descriptive counts across dependent, partly adaptively reused historical intervals, not independent statistical trials or a significance test. The 2025 weakness must be visible in the presentation.

## Final showable conclusion

A small InternVL3-2B RL adapter learned a more defensive mixture of causal allocation strategies. In the reserved evaluation it reduced volatility, drawdown, turnover and fees, improved observed Sharpe, and slightly reduced net return. Across additional historical regimes, the return and Sharpe benefits were inconsistent. Classical minimum variance remains a strong and sometimes better alternative. Generated portfolio explanations are a separate model capability with documented factual failures. These results support a transparent adaptation study, not a profit-improvement, universal superiority or reliable autonomous-adviser claim.


## Replacement headline comparison figure

Use final_presentation_figures/clean_final_comparison/trained_vs_untrained.png (or vector SVG/PDF) as the primary trained-versus-untrained figure. It supersedes the earlier cluttered multipolicy figure for the presentation headline. Four metrics, two consistent colors, direct values and signed changes isolate the final selected seed42 adapter on the reserved2026 interval. The conclusion is lower observed risk and higher Sharpe, slightly lower net return. CAPTION.md supplies the full evaluation and uncertainty statement. Keep minimum-variance/equal-weight baselines and annual robustness checks in supporting results, not omitted from the research paper. Earlier figures remain historical/supporting artifacts.


Publication figure revision: the primary trained_vs_untrained figure now has only four metric panels, axes, a shared legend and numeric values. Headline, directional guidance, change annotations and takeaway prose are removed. Use CAPTION.md for interpretation, evaluation dates, seed and uncertainty.


Publication figure revision: the primary trained_vs_untrained figure now has only four metric panels, axes, a shared legend and numeric values. Headline, directional guidance, change annotations and takeaway prose are removed. Use CAPTION.md for interpretation, evaluation dates, seed and uncertainty.


## Buy-and-hold baseline added

The preferred final comparison is now final_presentation_figures/clean_final_comparison/trained_untrained_buyhold.png (also SVG/PDF). It retains the untrained and trained seed42 policies and adds an equal-weight, initial-purchase-only baseline on the same five ETFs and exact189 return dates. One initial purchase at30January2026 costs $99.90; thereafter units remain fixed, with no rebalancing or terminal liquidation. Vendor-adjusted prices reflect dividends on the same basis as existing results. This is not the previously reported periodically rebalanced equal-weight comparator. Buy-and-hold net return 6.916245%, Sharpe 1.250403, max drawdown 4.449750%, annualized volatility 7.348328%. Source hashes, fixed units and daily values are in buy_and_hold.json; caption is BUYHOLD_CAPTION.md. No adapter or original financial bundle was modified. Endpoint-price wealth identity and absence of subsequent turnover were checked.


---

# Current presentation result

Use nifty_test_results/FINAL_TEST_REPORT.md and presentation/nifty_test_2025.*. Exclude development scores from the main presentation. The verified Nifty adapter shows training benefit versus raw on retrospective tests; its small buy-and-hold advantage is not statistically established. The running UI still uses the earlier ETF adapter; do not present its records as this Nifty result.
