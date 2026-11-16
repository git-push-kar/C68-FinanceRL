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
