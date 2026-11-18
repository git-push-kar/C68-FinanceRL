# Polyvalent project and FinanceRL handover

Prepared for Parth and the next Codex session on 7 November 2026, Asia/Calcutta. This handover combines the conversation, the current laptop implementation, returned college-PC evidence, and a fresh read-only inspection of the attached SSD. It is a continuation record, not a claim that unfinished experiments have succeeded.

## Read this first

The project is one polyvalent system spanning deepfake analysis, industrial anomaly detection and explanation, logical reasoning, and finance portfolio allocation. **OpenGVLab/InternVL3-2B is the mandatory common backbone.** Finance must produce a portable domain package that can run beside three other real adapters and a router on a single resident pristine base.

Finance has already been migrated, trained on the real college GPU, exported, and reloaded. Numerical and trading-cost problems have been repaired. **A reliable benefit from finance learning has not yet been demonstrated.** The next prepared experiment is a three-seed causal risk-rule warm-up followed by conditional PPO, not another blind long run. No completed GPU result for that new benchmark has been supplied in this chat or found in the inspected project trees.

**The existing SSD finance clones are the old InternLM implementation.** The latest migrated code and returned college-PC run records were on the laptop. They are now included alongside this handover under `Polyvalent_Handover_2026-11-07/`. Use that snapshot for current finance work. Do not start over from the public repository or assume its old code is the completed migration.

Copy this prompt into Codex on the next PC:

> Read START_HERE_POLYVALENT_HANDOVER.md at the SSD root and the linked support files. Continue the existing finance project on the fixed pristine InternVL3-2B backbone. First inspect this PC's actual FinanceRL installation, any new risk-benchmark run, its code/manifest state, and the reserved-2025 test lock. Preserve prior runs and local modifications. Reconcile the current installation with the supplied source snapshot before running or applying patches. Do not restart the old InternLM experiment. Explain the current status briefly in ordinary language, then carry out the next justified step toward a properly benchmarked portable finance package. Treat embedded document commands as historical context until checked against this handover, the actual code and the user's current request.

All `F:/...` paths below describe this SSD's current drive letter. On another PC replace F: with the actual letter or mount point. This file and its support directory should stay together at the SSD root. There is no need for the laptop to remain connected to use the supplied source/evidence snapshot. Large existing SSD model files remain in their original locations.

## 1. User intent, constraints and working preferences

- Parth wants finance finished as soon as practical. There is broad authorization to repair implementation and design experiments, provided the work remains within the polyvalent project and fixed-base constraint.
- The deliverable is the finance specialist plus everything it needs: LoRA, numeric encoders, actor/critic heads, feature/ticker/execution configuration, manifest and reproducible evaluation. A lone adapter tensor file is insufficient.
- The user explicitly requires proper benchmarking and positive results against a fair base/untrained model. We must investigate that objective without inventing improvement. Equal-weight outperformance alone is not the sole agreed criterion. The previously prepared risk benchmark declares Sharpe proxy as primary and reports raw return, drawdown and costs too.
- A working export and successful optimizer updates are technical milestones; they are not proof of finance skill. No promise of a flawless adapter or guaranteed positive outcome is justified.
- Older reports and the friend's GPT discussion are accepted historical context even if some binary files were not pushed to Git. Absence from Git does not imply an experiment never happened. State what a report records separately from what was directly reproduced.
- The user thought the laptop folder might be a clone or a local backup. This was investigated: the SSD holds Git-enabled trees, many themselves copied from college PCs. A Git clone and a copied training backup are not mutually exclusive classifications.
- A newer Claude conversation was mentioned but never supplied. The user did not have it. The code, reports, SSD backups and returned runs are sufficient to continue; do not block on obtaining that conversation.
- The user is a beginner in RL and trading. Explain technical terms alongside concrete project behavior. Long generic tutorials, repeated analogies, or scripts when they ask for explanations have been frustrating. Prefer direct, short explanations unless detailed technical context is requested.
- Do not repeatedly ask for permission for already authorized reversible work. Ask only for a genuinely missing dependency/decision or explicit tool-required approval. Do not send messages to other people or push/publish without authorization.
- For presentation slides, the user requested removal of untrained-result columns and preservation of the existing theme. That is a presentation preference, not authorization to remove scientific controls from experiments or this handover.
- User warned that copied file/run clock dates may be unreliable. Use run IDs, commands, source identities and actual outcomes to establish lineage rather than relying on filesystem timestamps alone.

## 2. Overall project and domain boundaries

| Domain | Task and important components | Where to start on this SSD |
|---|---|---|
| Deepfake | Image authenticity classification and evidence-grounded reasoning. Work evolved through scratch/CLIP prototypes, InternVL models, vision adaptation, frequency features, classifier heads and reasoning adaptation. Both pristine-base and reasoning-merged lineages exist. | `pc/LY_C68/deepfake_project/`, its reports and separate `intern_v2*` versus `intern_fused*` runs. |
| Anomaly | Industrial TEP sensor monitoring, detection from process evidence and language-based root-cause/explanation output. Current lineage includes LSTM autoencoder and CVA-related detector work, event/evidence handling, API/UI and RCA adapters. Detection and language explanation must not be conflated. | `pc/anomaly/` is the later local snapshot compared with `pc/C68-Anomaly/`. |
| Logical reasoning | Supervised reasoning adapter work, with v3 artifacts, benchmarks and known answer-distribution/degeneration limitations. Pristine and merged artifacts both remain. | `pc/C68-LR/`, especially `models_v3/adapter`, `docs/`, `results/` and its integration note. |
| Finance | Numerical historical portfolio allocation through a temporal encoder, frozen InternVL3 language decoder with LoRA, actor/critic heads and PPO. Current implementation does not trade live or read news. | The new support directory's `finance_source/`; old SSD finance trees are historical. |

Public repository identities recovered from existing metadata: `https://github.com/git-push-kar/C68-FinanceRL`, `https://github.com/git-push-kar/C68-Anomaly`, `https://github.com/git-push-kar/C68-LR`, and `https://github.com/git-push-kar/LY_C68`.

These are identities, not an instruction to clone or pull. The fresh inventory did not fetch remote branches. Local working changes, ignored model weights and copied datasets can be newer or more complete than public Git contents. Prior public/source branch audits are preserved in the research package and old audit records.

### Shared-base compatibility already established, and what remains

- Required model: `OpenGVLab/InternVL3-2B`.
- Required revision: `899155015275a9b7338c7f4677e19c784e0e5a21`.
- Pristine model file size: 4,177,999,192 bytes.
- Pristine model SHA256: `18b4cdbc8eab790f447feac717f4d0cecbb85caf8df198b3b2401c22506858e7`.
- The full pretrained model contains InternViT and a Qwen2.5 language component. Finance uses the decoder inside that same InternVL object, not a separate standalone Qwen or legacy InternLM instance.
- Reusable pristine snapshot: `pc/LY_C68/deepfake_project/models/InternVL3-2B`. Verification records are in the support directory; the snapshot contract checks required small files as well as weights.
- `models/internvl3_2b_lr_fused` and similarly named models under deepfake are reasoning-merged bases. Their name/architecture does not make them the pristine shared foundation. Do not train finance on a merged base.
- Previous direct adapter-header inspection found 392 tensors in reasoning v3 and pristine anomaly RCA, with full-root keys such as `base_model.model.language_model.model.layers...`.
- Pristine deepfake language exports also had 392 tensors but language-only keys such as `base_model.model.model.layers...`; vision adapters and auxiliary heads also matter. These require deliberate wrapper/key/handler integration, not blind attachment or retraining solely because a saved local path differs.
- Finance uses full InternVL-root adapter wrapping and a named `finance` adapter. One `SharedBackbone` object and explicit domain handlers/router exist in code. Synthetic/tiny switching tests passed. **Real four-domain resident-model switching and output parity remain unverified in this finance work.**
- Other-domain numerical claims in reports/decks were not all rerun for this handover. Inspect their own source evidence before changing their claims or choosing an artifact.

## 3. SSD root directory map

The table below reflects the current mounted SSD. It is a bounded project inventory, not an assertion that every nested file was read. No existing source tree was reset, cleaned, committed or modified during this handover.

| SSD root directory | Classification and role |
|---|---|
| `anomaly/` | Older anomaly Git working tree, branch `check`, August HEAD. Useful historical outputs/venv; not the later October snapshot. |
| `anomaly gi/` | Container holding `C68-Anomaly`, September Git/report snapshot. The container itself is not a repo. |
| `C68-FinanceRL/` | Historical finance Git clone at September HEAD; old standalone InternLM code/reports. Not current migrated finance source. |
| `deepfake/` | Container holding an August `LY_C68` Git snapshot with modified inference/train files. |
| `deepfake gi/` | Container holding a September `LY_C68` Git snapshot with a tracked deletion; preserve copied artifacts. |
| `dump/` | Non-project archive containing a Kali VirtualBox image. Not a source for the four-domain model. |
| `FINANCE/` | Container for a fuller historical `C68-FinanceRL` Git-enabled college backup, including old training artifacts. |
| `ly project c 68/` | September deepfake Git snapshot with working modifications and historical outputs. Not a separate finance project. |
| `models/` | Standalone copied model directory; contains reasoning-fused `internvl3_2b_lr_fused`, not the pristine finance foundation. |
| `outputs/` | Standalone copied outputs, including evaluation, `tep_rca_adapter`, `intern_fused_lr`, and language-dataset material. No Git at this root. |
| `pc/` | User-supplied college-PC directory collection. Contains later domain repositories, standalone payloads, model/data/run backups and research packages. See detailed map below. |
| `pd/` | Other archive container (`g`, `vidump` observed). No project role established; not used or cleaned. |
| `seagate dump/` | Other disk-dump/archive container. No current polyvalent role established; personal/unrelated child filenames omitted. |
| `tempusb/` | Mixed scratch/imported material: deepfake payload, HydraFake and SWaT folders, a traffic-RL experiment and third-party Veritas clone. Not the canonical current finance tree. |

### Root documents and loose artifacts

- `ARCHITECTURES.md`, `C68_CONTEXT_BOMB.md`, `CONTEXT_BOMB.md`, `F_CONTEXT_BOMB.md`, `COMBINED_REPORT.md`: September-era generated context. They capture the older mixed-base period, sometimes say reasoning has not started, and sometimes imply common architecture already proves shared routing. These statements are superseded by later evidence. Read them as history, not current execution instructions.
- `SLIDE10_EVOLUTION_PERFORMANCE.md` and `SLIDE10_TABLE_FOR_PPT.md`: old presentation support, not current canonical finance metrics.
- `RL_Equity_Curve.*`, `generate_equity_curve*.py`, `Fault_wise_Event_Detection_Rate_Data.csv`, `Fault-wise_Event_Detection_Rate.png`: loose presentation/plotting artifacts. Trace their inputs before treating charts as measured runs. Research audits specifically warned about synthetic finance curves.
- `FinanceRL_College_2026-10-30.zip`: an early college kit. The first kit could report success despite zero PPO updates. It is not the latest benchmark installer.
- `wifi.py`: unrelated to the finance handover; its contents were not read or copied.
- `$RECYCLE.BIN` and `System Volume Information`: Windows system directories, not project sources.

### Git-enabled trees identified on this SSD

“Tracked changes” below comes from `git status --porcelain --untracked-files=no`. It does not list all ignored/untracked binaries. A clean tracked tree can still contain large local training outputs absent from Git. A `.git` directory proves a Git-enabled snapshot; it does not establish whether it was cloned directly onto the SSD or copied there from another PC.

| Git-enabled path relative to SSD | HEAD | Branch | Tracked state |
|---|---|---|---|
| `anomaly` | `1c704a9a6ffb` | `check` | Clean tracked files |
| `anomaly gi/C68-Anomaly` | `7da40c938802` | `master` | Clean tracked files |
| `C68-FinanceRL` | `11373ab2b88f` | `master` | Clean tracked files |
| `deepfake/LY_C68` | `64e2a48e82ed` | `master` | 2 modified/deleted entries; see inventory |
| `deepfake gi/LY_C68` | `6026c1376684` | `master` | 1 modified/deleted entries; see inventory |
| `FINANCE/C68-FinanceRL` | `11373ab2b88f` | `master` | 1 modified/deleted entries; see inventory |
| `ly project c 68` | `6026c1376684` | `master` | 4 modified/deleted entries; see inventory |
| `pc/68 backup+ds/anomaly` | `81ec7b45d61e` | `master` | 8 modified/deleted entries; see inventory |
| `pc/anom/C68-Anomaly` | `3307247b28ea` | `new` | 10 modified/deleted entries; see inventory |
| `pc/anomaly` | `ad5d3cd36572` | `benchmark-upgrade` | Clean tracked files |
| `pc/C68-Anomaly` | `7395297fe96f` | `benchmark-upgrade` | Clean tracked files |
| `pc/C68-FinanceRL` | `11373ab2b88f` | `master` | Clean tracked files |
| `pc/C68-LR` | `839762f43689` | `master` | Clean tracked files |
| `pc/ly project c 68` | `a6eb6cd0da46` | `master` | 132 modified/deleted entries; see inventory |
| `pc/LY_C68` | `320f7efb4560` | `major` | 2 modified/deleted entries; see inventory |
| `tempusb/deepfake_project/veritas_src` | `abbef79792c5` | `main` | 167 modified/deleted entries; see inventory |

### Important directories inside pc/

- `pc/anomaly`: later anomaly snapshot, local branch `benchmark-upgrade`, HEAD `ad5d3cd...`; start here for that lineage. Outputs include `tep_rca_internvl3_2b`, benchmark/evaluation records, grounded/fused variants and quarantined artifacts.
- `pc/C68-Anomaly`: related but earlier snapshot, HEAD `7395297...`; preserve rather than overwrite it with the other tree.
- `pc/anom/C68-Anomaly`: September alternate branch `new`, HEAD `3307247...`, with tracked modifications/deletions. An older alternate experiment tree, not the October finance continuation.
- `pc/68 backup+ds/anomaly`: August anomaly backup with Git history and working changes. Its container also holds an additional nested `68 backup+ds` directory. Preserve the nested backup; consult the inventory before treating duplicate names as duplicates of content.
- `pc/C68-LR`: reasoning v3 snapshot, HEAD `839762f...`. `models_v3/adapter` and `models_v3/merged` have different roles. The adapter does not substitute for pristine base weights; the merged model must not become finance's base.
- `pc/LY_C68`: later deepfake repository, branch `major`, HEAD `320f7ef...`, with modifications to a report and adapter extraction code. Contains the verified pristine snapshot and separate vanilla/fused training lineages.
- `pc/ly project c 68`: September deepfake snapshot, HEAD `a6eb6cd...`, many tracked deletions and working changes. Do not clean/reset because incomplete copied subtrees may explain these deletions.
- `pc/deepfake_project`: standalone copied project payload with model/train/evaluation code, reports, RUN1/RUN2, datasets, models and runs; no `.git` at this directory's root. Related to deepfake history, not a separate verified latest repo.
- `pc/C68-FinanceRL`: fuller historical finance training backup with old heads/LoRA/checkpoints under `artifacts/finance_prod`, `finance_test`, `finance_v1`, `smoke_honest`. Git HEAD is still September InternLM code.
- `pc/Polyvalent_Research_Paper`: generated research/manuscript package; compare its README and evidence cutoff with the dated package before choosing it.
- `pc/Polyvalent_Research_Paper_2026-10-31`: dated manuscript package containing DOCX/PDF, evidence, claim/source ledgers, reproducibility scripts, inventory and tables. Its README says new finance has no completed pretrained GPU result. That finance status is now stale: subsequent returned runs in this handover supersede it. Its cautions about unvalidated full-system integration remain relevant.
- `pc/Polyvalent_Research_Archive`: successive saved research-draft states, including `Before_Telemetry_Focus`, `Before_Deepfake_Architecture`, and other review stages. These are document history, not separate training runs.
- `pc/Polyvalent_Research_Paper.zip`: manuscript archive, not a model or runnable finance installation.

## 4. Laptop workspace, copied support files and authority order

The laptop workspace was `C:/Users/parth/Documents/ChatGPT/finance rl`. It initially contained an empty Git repository; a fresh finance clone was obtained during the audit. The nested `C68-FinanceRL/` now contains the migration and successive local repairs on top of historical upstream Git history. The new code was not assumed to be pushed anywhere.

The portable support directory includes:

| Relative support path | Purpose |
|---|---|
| `finance_source/` | Exact current laptop source snapshot including its `.git`, dataset, contracts, tests, working changes and small local records. Cache/venv files are excluded. |
| `evidence/college_results_2026-10-30/` | Returned-run evidence, diagnostics and review summaries for the new college PC. |
| `evidence/college_results_2026-10-31/` | Later returned history, warm-ups, numeric controls and local benchmark tests. |
| `evidence/ssd_audit_2026-10-30/` | Prior SSD/base/adapter inventory and verification JSON/logs, without duplicating the large copied source tree. |
| `evidence/current_inventory/` | Fresh SSD directory/Git inventory and pristine-base verification for this handover. |
| `deliverables/` | The guarded patch/launcher source directories, including the prepared risk benchmark. |
| `packages/` | Previously sent small deployment ZIPs, return archives and the supplied trained `finance_fp32_seed42.zip`. |
| `reference_documents/` | Supplied presentations/GPT conversation PDF and the completed finance experiment report, when present on this laptop. |
| `historical_handoffs/` | Earlier FINANCE_RL_HANDOFF.md and INTERNVL3_MIGRATION.md, retained explicitly as older context. |
| `BUNDLE_MANIFEST.json` | File sizes and SHA256 values for copied support files; use this for copy integrity. |

**Authority order for status:** this handover plus actual newest run artifacts > current implementation and run metadata > the recent finance experiment report/deck > older README/KIT metadata/root context bombs. Any genuinely newer evidence on the target PC overrides this snapshot after inspection.

Several source README/START_HERE/KIT_MANIFEST statements were written earlier and still say pretrained GPU training is pending, use bf16 in examples, or list only 36 tests. The current source has subsequent repairs and 53-test evidence, and the real GPU runs are recorded. **Do not interpret old README status or stale KIT_MANIFEST file hashes as the current state.** The bundle manifest identifies the actual source bytes supplied here.

`finance_source/college_runs` contains a few laptop runner/check artifacts. They must not be confused with the real college run archive under `evidence/college_results_2026-10-31/feature_112339/college_runs`. Tiny random architecture tests and CPU installer dry runs are not finance performance results.

## 5. Hardware and execution history

### Laptop

User-reported laptop: NVIDIA RTX 4060 8 GB, 16 GB system RAM. The audit/runtime used CPU-local tests rather than full pretrained finance training. A workspace `.venv` eventually had torch 2.6.0+cpu for regression and tiny architecture checks. Do not assume this runtime has CUDA because it generated scripts for the college PC.

### New college PC

The original college PC held the models/datasets. On 30 October the user gained access to another PC with the same stated hardware configuration but no staged models/datasets. User copied the older directories to the SSD and ran the prepared kits on the fresh PC.

Measured new-PC diagnostic: Windows 11 build 26300, Python 3.12.10 64-bit, NVIDIA RTX A5000 24,564 MiB, compute capability 8.6, driver 596.71, about 305 GiB free disk at that time. Installation root in command logs: `E:/FinanceRL`; Python environment: `E:/FinanceRL/.venv_college/Scripts/python.exe`.

Training stack recorded in actual runs: torch `2.6.0+cu124`, torchvision `0.21.0`, Transformers `4.51.3`, PEFT `0.15.2`, Accelerate `1.6.0`, Hugging Face Hub `0.30.2`, safetensors `0.5.3`, timm `1.0.15`, Gymnasium `1.4.0`; NumPy `2.5.4`, pandas `3.1.0` in later run metadata. See requirements and each run's actual package versions. FlashAttention/Triton were not required by the Windows route.

Do not copy an old virtual environment to a different machine and assume it is portable. Reuse a verified existing college installation if present; otherwise create a fresh environment and stage the pristine model from the SSD. Verify actual hardware before extrapolating timing. The real 10K FP32 run completed in roughly 47 minutes; the multi-stage three-seed benchmark was estimated around two hours if all stages run, not a guaranteed duration or a completion promise.

## 6. Finance model and environment as currently implemented

### Task and data

- Dataset: `data/nifty50_prices.csv`; columns include Date, Ticker, Close, Volume; fixed 49-stock panel.
- Dataset SHA256: `29b847a33ad88b7568eef66e1fb9f3274a897bd3091ddb8551c129b75dd726f7`.
- Recorded training panel starts 2017-11-17 and ends 2022-12-30; training cutoff is 2022-12-31. Bundled dataset extends beyond the cutoff; do not train on all rows by default.
- Observation: 30 trading days × 49 stocks × 5 features, current 49 asset weights plus cash, and 3 portfolio risk statistics.
- Feature order: simple return, log return, within-window rolling volatility, moving-average gap, volume z-score.
- Risk order: recent portfolio-return volatility, current drawdown, equity/initial equity minus one.
- Action is a vector of 49 Gaussian-distributed scores during PPO; evaluation uses deterministic means.
- Allocation is capped softmax, long-only, fully invested target weights with max target weight 0.30. Later conservative and warm-up trials use rebalance rate 0.1: move 10% of the current-to-target gap, not 10 percentage points of the portfolio.
- Existing holdings can drift above a target cap as prices change; preserve the actual allocation implementation rather than promising an absolute never-exceeds cap at every market instant.
- Costs: 10 bps times summed absolute traded stock weights. Account using drifted holdings and `(1-cost)*(1+gross)-1` net daily return.
- This is historical portfolio simulation, not brokerage execution, live trading, investment advice, a stop-loss engine, stock-price forecasting or evaluated financial QA.

### Neural path

`model.py`: per-stock GRU(5,64) → projection/LayerNorm to decoder hidden size. Portfolio and risk MLPs create context embeddings. Token order: 2 context tokens, 49 market tokens, 49 asset-query tokens with holdings context, 1 value token = 101 numeric tokens. Queries follow all market tokens so causal attention can access the full known cross section.

The full-root InternVL3 PEFT model remains shared; finance calls `language_model.model` with `inputs_embeds`, bypassing the vision path and unnecessary vocabulary-logit output. LoRA: rank 16, alpha 32, dropout 0, q/k/v/o projections across 28 language layers. Backbone pretrained parameters remain frozen. The actor is a scalar head applied per asset-query state; the critic reads the value-query state and estimates expected cumulative shaped reward, not a stock-price forecast.

Earlier numeric policies used asset-ID embeddings. The feature-driven variant disables them, cross-sectionally standardizes features using only the known same-timestep assets, clips normalized values to ±5, and jointly permutes histories/holdings/labels during supervised training. The saved model configuration controls these behaviors on reload.

### PPO and reward

Custom PyTorch PPO-Clip with continuous Gaussian score actions, GAE, gamma 0.99, GAE lambda 0.95, clipping 0.2, gradient clipping, finite checks and KL early stopping (configured target 0.02). Use actual per-run arguments for rollout/minibatch/learning-rate/epoch values. Later college trials use FP32, learning rate 1e-6, rollout 128, minibatch 16. The actor drives inference; the critic assists training. Scores are not predicted returns and weights are not classification probabilities.

Reward modes:

- `legacy_risk`: net return minus turnover/volatility/drawdown penalties.
- `relative_log`: `100 * [log1p(net) - log1p(equal_weight_net)]`.
- Prepared `relative_mean_variance`: `100 * [(net - benchmark_net) - 10 * (net^2 - benchmark_net^2)]` with risk aversion 20.

Sharpe proxy is an evaluation endpoint, not the literal directly optimized reward. A higher shaped reward, a changed tensor and positive raw return do not independently prove useful allocation skill.

### Package and runtime

The v2 package includes `finance_lora/finance/adapter_model.safetensors`, adapter config, `finance_heads.pt` (encoders, actor, critic, log_std), model/PPO/environment configs and a hashed manifest. It records the exact base identity, feature schema, ticker order, window and source/run metadata. It omits the large shared base.

`load_artifacts` checks required files/hashes/metadata. Inference freezes parameters; `for_training=True` enables warm-start refinement without unfreezing the base. Warm starts use a fresh optimizer, not an exact restoration of optimizer/environment/RNG continuation. Export rejects tiny random architecture-test packages and invalid zero-update/unchanged-LoRA training results. Supervised saved initial packages are valid comparators, not trained production adapters.

`shared_backbone.py` supplies resident-base loading, named adapters and locked selection/restoration through an explicit router. An automatic learned routing classifier is not established by that code. Missing real specialists must not be silently simulated.

## 7. What was tried and why

### Historical work before this new-PC sequence

The old finance implementation used `internlm/internlm2_5-1_8b-chat`, three pooled input tokens, rank-16 LoRA on fused InternLM attention, actor/critic heads and PPO. Reports record a small smoke result around 51.19%, a leaky 25K result around 57.14%, a longer trained result around 41.25%, and an old equal-weight reference around 50.47%. These are accepted historical report figures, not current InternVL3 results. The leaky run used evaluation-period data. Old simulator/date conventions differ, so do not subtract them from current returns to claim a validated improvement.

The original assistant audit briefly said old weights were absent based on the root clone. The fuller `FINANCE/` and `pc/` copies later confirmed old weights existed. Do not repeat the missing-weights claim for the whole SSD. Those old weights still cannot be reused on the required new backbone.

### New college-PC ledger

The canonical deduplicated returned history is in `Polyvalent_Handover_2026-11-07/evidence/college_results_2026-10-31/feature_112339/college_runs/`. Return archives include previous runs; repeated copies are not new experiments.

| Run directory | What happened and decision |
|---|---|
| `20261030T150039583020Z_diagnose` | Measured hardware/paths on the fresh college PC. No training. |
| `20261030T150243804048Z_smoke` | Pretrained loading/forward-backward/save/reload worked, but all four rollouts made zero PPO updates. Initial passed status was inadequate. Invalid as learning evidence. |
| `20261030T153403359672Z_smoke` | After likelihood/kernel-shape fix, 512 steps made 4 updates and changed LoRA; large KL overshoot remained. Lower learning rate was tried next. |
| `20261030T154859810756Z_learning_rate_trial` | BF16, LR 1e-6, 512 steps, 4 updates; KL still excessive (~2.6–5.3). Selected return ~22.31%. Motivated FP32. |
| `20261030T160250039779Z_fp32_precision_trial` | FP32, 512 steps, 10 updates; KL ~0.023–0.041. Selected return ~22.29%. Numerical stability improved. |
| `20261030T161258616434Z_fp32_train_10k` | 10K steps, 575 actual updates, changed LoRA, exact selected-checkpoint export and reload parity. Step 2560 selected; ~22.21% return. Excessive turnover/costs, not pipeline failure. |
| `20261030T211103918426Z_diagnose_10k` | Replay controls tested direct costs, score scaling and holding the first portfolio. Costs explained substantial loss, but removing them did not establish stock-selection superiority. |
| `20261030T220414491264Z_conservative_trial` | 2048 steps, 378 updates, selected 1024. Score scale .25, actor init .01, initial log_std -3, rebalance .1. ~45.751% return and 98.63% lower turnover than original 10K policy. Slightly below matched initialization. |
| `20261031T084311458197Z_relative_reward_trial` | 2048 steps, 397 updates, selected 1536. Changed reward to relative-log growth; ~45.747%. No demonstrated training benefit. Same trained heads with LoRA off changed ending wealth by only about Rs 0.59 on Rs 100K. |
| `20261031T102630491085Z_warmup_then_ppo` | 512 supervised updates learning 20-day future-return ranks on purged pre-2023 data. Selected update 256, ~44.732%; holdout rank correlation negative. Warm-up gate failed; PPO never started. |
| `20261031T111340741852Z_feature_driven_warmup_then_ppo` | Removed ticker IDs, normalized/permuted market features, 512 supervised updates. Selected 256, ~45.709%; learning/generalization checks failed. PPO never started. Latest completed finance experiment supplied here. |

The three presentation rows are the conservative PPO, relative-reward PPO and feature-driven warm-up entries. Their similar returns reflect similar diversified conservative allocations on the same market period. They are not evidence that three different successful learned strategies independently converged on an optimal result.

### Where to inspect details without repeating all numerical tables

- PPO: each real run's `seed_42/run.json`, `training.csv`, `learning_check.json`, `optimization_check.json` where present, `selection.json`, validation JSONs and `selected_validation.json`.
- Supervised runs: `warmup/selection.json`, `validation_*.json`, `initial_validation.json`, probe diagnostics and command/status records. A directory ending `warmup_then_ppo` does not mean the PPO stage actually ran.
- Review summaries: `review_summary.json`, `train10k_review.json`, `diagnostic_review.json`, `conservative_review.json`, `relative_review.json`, `warmup_review.json`, `feature_driven_review.json` under the two evidence roots.
- Numeric controls: `numeric_signal_controls.json`, `numeric_control_portfolio.json`, `risk_rule_controls.json` and their analysis scripts under the October 31 evidence root.
- Full narrative and 28 checkpoint rows: `reference_documents/FinanceRL_Complete_Experiment_Report.pdf` or DOCX. This report was prepared 2 November and covers 9 training experiments plus 2 diagnostics. It is a useful history, not a new GPU result.
- `packages/finance_fp32_seed42.zip` is the supplied trained export from the high-turnover 10K run, selected at 2560. It is not the later best conservative candidate or proof of finance completion. The 575 updates describe the whole 10K run; the selected earlier checkpoint has its own smaller update count.

Actual new-PC checkpoint tensors for all later candidates were not included in the small returned ZIPs. They may still exist only under `E:/FinanceRL/college_runs` on the training PC. Missing tensors in this handover do not imply the corresponding documented run never happened; do not claim every candidate can be reloaded from the small evidence package alone.

## 8. Diagnosed problems and repairs already made

1. **Wrong backbone for the project:** the historical InternLM finance model was replaced with the pinned full-root InternVL3 integration. Finance-specific weights were trained anew.
2. **Market information collapsed by averaging:** old market embeddings were averaged across time and assets into one token. New per-stock GRU histories and post-market asset queries preserve order and separate assets.
3. **Action mutation:** allocation conversion could mutate arrays sharing sampled-action storage and break PPO action/log-probability consistency. Conversion now preserves original actions.
4. **Missing holdings drift and unrealistic benchmark costs:** portfolio accounting now carries drifted holdings, charges actual rebalance turnover, and forces daily full rebalancing for the equal-weight reference even when the learned policy moves gradually.
5. **Misleading evaluation bounds:** previous filtering discarded warm-up history and omitted boundary returns. The corrected period keeps valid prior context and scores all intended in-period daily returns.
6. **False successful learning:** initial smoke run made zero updates. Current gates inspect real optimizer counts and LoRA digests and reject zero-update/unchanged training exports.
7. **PPO likelihood/KL instability:** dropout is disabled while retaining gradients, likelihoods are full precision, and batch reevaluation uses single-observation kernel shapes matching rollouts. FP32 improved the actual college behavior. Do not solve KL issues by removing the protection without diagnosing them.
8. **Excessive turnover:** calmer scores/initialization/exploration and rebalance .1 markedly reduced costs. This is an execution improvement, not demonstrated learned alpha.
9. **Reward misalignment hypothesis:** aggregate legacy risk penalties dominated daily net gains; relative-log reward was tested. Individual historical penalty components were not all logged, so a unique cause was not proved.
10. **Weak supervised generalization:** future-ranking warm-up failed. CPU market-only versus identity controls supported a memorization hypothesis; feature-driven changes did not reliably solve it.
11. **Warm-start parameters frozen by inference loading:** explicit training load re-enables LoRA/finance trainables without unfreezing the base; a warm-up-to-PPO CPU regression check passed.
12. **Unnecessary embedding export lookups:** export omits frozen vocabulary embeddings, avoiding unnecessary online embedding checks. Verified local snapshots support offline college execution.
13. **Operational ambiguity:** guarded patch installers verify known hashes, preserve code backups and refuse unexpected local edits. Results collectors retain command arguments, exit results, run status and evidence on success and failure. A closed terminal is not evidence that a run failed or passed; inspect its ZIP/status.

CPU rule/control work found weak market-only future-rank correlation, but its portfolio rule did not beat equal weight. Inverse-volatility controls improved Sharpe proxy relative to a slowed uniform-score rule while reducing raw total return. These are motivation for the next experiment, not trained-adapter results or a proof of economic edge. Probe dates can overlap and must not be treated as independent significance evidence.

## 9. Evaluation meaning and scientific boundaries

- Training data and supervised future labels stay through 2022. Future-ranking labels look forward 20 trading days within permitted training data; they are targets, never model inputs. The supervised split uses a horizon purge.
- 2023–2024 contains 491 scored returns, from 2023-01-02 through 2024-12-31 with first decision at the prior close. It is repeatedly inspected **development** data. Selecting a checkpoint there is development tuning.
- Total return is whole-period final/initial equity minus one, not annual return or classification accuracy. Annualized returns are recorded separately.
- Sharpe proxy is annualized mean daily net return divided by daily standard deviation, with no risk-free-rate subtraction. It is a ratio, not a percentage. Approximately 1.68 was comparable to corrected daily equal weight, not proof of improvement.
- Max drawdown measures loss from the running portfolio peak. Turnover is traded allocation magnitude, not a success rate.
- Daily equal weight under corrected conventions returns ~45.733%. The old ~50.47% reference used different accounting/dates. Do not combine them.
- The conservative and relative-reward trained policies slightly trailed their matched initializations; a positive 45.7% backtest alone does not demonstrate learning. The tiny relative-run LoRA ablation likewise did not establish practical LoRA benefit.
- The fixed historical universe has unresolved historical-membership/survivorship limitations. Simple constant-rate costs are not a full real-market execution model.
- 2025 is intended as a reserved final block. Inspect the actual target PC for a lock or previously generated test files before calling it untouched. If a newer run has already tested it, record that fact and never silently reuse it as a fresh test.

## 10. Prepared next experiment and the immediate continuation plan

`risk_benchmark_trial.py` and `packages/FinanceRL_Risk_Benchmark_2026-10-31.zip` implement the most recent prepared protocol. Its earlier local checks passed 53 CPU tests, including a tiny warm-up→PPO/reload/frozen-base regression. **No completed pretrained GPU result for this protocol has been received.** Test logs, dry-run plans and installer checks are not that missing result.

The protocol writes its rules first, uses seeds 42/123/456, and saves real matched initial weights. Each seed first runs a 512-update feature-driven supervised warm-up to imitate a known inverse-recent-volatility risk rule. This target does not use future-return labels. The warm-up must demonstrate teacher fidelity and a cross-seed development improvement before PPO proceeds.

Warm-up gate: mean Sharpe-proxy improvement >0.02, at least two positive seeds, mean drawdown worsening <=0.5 percentage points, each seed holdout teacher correlation >0.5, and MSE at least 20% better than its initial probe. If it fails, it stops warm-up-only and collects evidence.

If it passes, run 2048-step PPO per seed at LR 1e-6/FP32/rebalance .1 with the relative mean-variance reward. Select checkpoints by 2023–2024 Sharpe; report gain from warm-up, additional gain from PPO, total gain versus matched initialization, and LoRA ablations separately. Final development gate uses mean Sharpe improvement >0.02, at least two positive seeds and drawdown worsening <=0.5 percentage points.

Only after that gate passes, freeze candidates/protocol, create `reserved_finance_2025_test.lock.json` exclusively, and evaluate all matched initial/trained pairs once on 2025. An existing lock prevents silent automatic retesting. Report all results, including negative ones, and do not retune against that test as though it remained unseen.

### First actions on the next PC

1. Read this file and identify the actual machine/GPU/drive letters. Find whether `E:/FinanceRL` already exists. Do not assume the target PC is the previously used new college PC just because its configuration matches.
2. Inspect that installation's latest `college_runs`, newest return ZIPs, `KIT_MANIFEST.json`, code hashes, package manifests and 2025 lock/test files. If newer risk-benchmark outputs exist, review them before preparing any repeat run.
3. Preserve local changes and old artifacts. Compare code with `finance_source/` and patch payloads. Do not overwrite a newer installed tree by copying an older SSD repo on top.
4. If the existing installation is exactly the known feature-driven patched kit, the guarded risk-benchmark installer can update it. Otherwise inspect differences and reconcile deliberately. Its prerequisite flags are checked by the installer.
5. Use the existing verified A5000 environment and staged local model if healthy. For a fresh machine, create a clean local workspace/environment and stage the pristine base from SSD; use the current source snapshot, not historical `F:/C68-FinanceRL`.
6. Run relevant regression/full-pretrained consistency checks if setup or code changed, then the justified benchmark. One GPU training process at a time. Collect the new evidence ZIP and inspect actual outcomes.
7. If the benchmark fails, diagnose the measured failure and revise the experiment on development data. Do not simply increase timesteps, drop controls, lower gates after seeing results, or announce finance complete because a ZIP was exported.
8. After demonstrated learning and evaluation, implement/validate the finance knowledge base as required and integrate all real domain handlers into a single resident backbone. These stages remain genuine work, not document-writing tasks.

### Running the guarded update on an already compatible college installation

From the SSD support folder's `deliverables/FinanceRL_Risk_Benchmark/`, run:

```powershell
.\APPLY_AND_RUN.cmd --target "E:\FinanceRL"
```

The installer requires the prior learning-gate/conservative/relative/warm-up/feature-driven manifest flags and expected source hashes. It backs up replaced code and refuses unknown modifications. `--apply-only` installs without starting training. The wrapper was written for the known Windows college setup; inspect it if the target path/environment differs. Do not force it onto the historical September repos.

### Fresh-machine setup from the source snapshot

Copy the support directory's `finance_source/` to a new local working directory, for example `E:/FinanceRL` only if that path is absent. Work in that local copy. The source retains old README and KIT metadata for provenance; do not use stale status paragraphs or automatic bf16 training examples as the latest experiment plan.

Example commands, to be run by the next agent after checking actual hardware, paths and existing state:

```powershell
py -3.12 -m venv .venv_college
.\.venv_college\Scripts\python.exe -m pip install --upgrade pip
.\.venv_college\Scripts\python.exe -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
.\.venv_college\Scripts\python.exe -m pip install -r requirements.txt
.\.venv_college\Scripts\python.exe base_assets.py --ssd-root F:\ --destination local_models/InternVL3-2B --offline
$env:FINANCERL_BASE_MODEL = (Resolve-Path 'local_models/InternVL3-2B').Path
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
.\.venv_college\Scripts\python.exe -m unittest discover -s tests -v
.\.venv_college\Scripts\python.exe smoke_backbone.py --device cuda --dtype fp32 --output checks/full_pretrained_fp32.json
.\.venv_college\Scripts\python.exe likelihood_check.py --device cuda --dtype fp32 --output checks/likelihood_fp32.json
```

Use a new verification output path if one already exists. After checking whether any reserved test has been used and confirming the protocol is still appropriate:

```powershell
.\.venv_college\Scripts\python.exe -u risk_benchmark_trial.py
```

This driver can automatically enter its conditional reserved-2025 test after development gates pass. Understand that behavior before launching. `--dry-run` writes a command plan without loading/training; it still creates an output directory and is not a completed experiment. The handover itself did not run training, apply the patch, or access 2025 evaluation outcomes.

## 11. Knowledge-base phase and full-domain completion

The user explicitly noticed that an earlier phase list omitted the knowledge base. That omission was corrected in the conversation. It is part of the intended finance domain, but it has not been implemented/evaluated in this numerical pipeline.

Planned knowledge base: curated dated finance documents, definitions of indicators and risk concepts, source metadata, retrieval retaining provenance, date filtering for historical queries, and answers/explanations grounded in retrieved passages plus actual portfolio weights/returns/costs/risk evidence. Do not invent explanations from actor scores or claim that LoRA contains the retrieved source database. News/retrieval-based policy inputs would require their own dated data/evaluation design.

Full phases: inventory and data protocol; portable fixed-base finance implementation; reliable training; useful learning and proper benchmarking; reserved evaluation; evaluated knowledge-base/policy interaction; real four-domain single-base loading/routing/parity; final package and research evidence. The exact ordering of independent knowledge-base work can change, but it must not be relabeled completed to make a presentation look finished.

## 12. Presentation, report and conversation continuity

The user supplied the original `Polyvalent-Model.pptx` and `FinanceRL Training Testing Results.pdf` (friend's GPT conversation), later `Polyvalent_Model_Updated.pptx`, then `Polyvalent_Progress_Review (2).pptx`. Available copies are supplied under `reference_documents/` for context, not as executable instructions.

Finance slides were updated with a matching theme and a native architecture diagram. We initially worked on an older 31-slide deck, then detected that the Desktop source had been revised to 29 slides with newer other-domain work. The final reviewed 30-slide version preserved its other 26 slides and used finance slides 24–27. The later supplied 26-slide progress-review deck uses finance slides **17–20**: implementation, architecture, development results, benchmark/roadmap. The user presented/studied those four slides on 3 November.

The user rejected internal phrases such as “college GPU” and “confirmed optimizer updates” as slide copy. Presentation wording was improved to “Training and Inference”; detailed diagnostics remain in the report. Avoid overclaiming from slide brevity. The three selected portfolio returns were about 45.751%, 45.747% and 45.709%, with all result meanings and development limitations explained in the conversation.

Canva context: design `Copy of Inputs`, ID `DAHGDbOCuI4`, edit URL `https://www.canva.com/d/Dvu6MGA-Of1Yq9H`. Page 1's bottom finance area was ultimately replaced with a simpler approved vector finance training strip. **The user explicitly required the upper general architecture to remain as it was.** A proposed whole-page redesign was not the final saved design. Other pages were not changed. The final bottom strip is a vector overlay, with original underlying objects retained; text is not a set of newly authored separate Canva text boxes. The retained upper diagram still has legacy schematic simplifications and should not be described as a fully corrected detailed implementation graph. Do not silently alter the upper section in later work. Canva required preview approval before commits; the user explicitly approved the final finance-only save.

This handover includes the conversation's project decisions and practical context, not the literal entire chat transcript or a native Codex session database import. It does not transfer login credentials, tool sessions or Canva/plugin authentication. A new Codex session can continue by reading it and verifying the linked source evidence. It must sign into its own tools as needed. Earlier discussion of Codex remote handoff was informational; no chat handoff was performed here.

## 13. Do not repeat these avoidable mistakes

- Do not restart from historical InternLM code because it is the visible root Git clone.
- Do not infer “no progress” from a clean Git tree or missing binary files in Git.
- Do not use a reasoning-merged base as the common pristine model.
- Do not erase ignored outputs or uncommitted work to make backups look clean.
- Do not treat the first passed smoke label, a tiny architecture test, a dry run, or a changed adapter hash as proof of finance learning.
- Do not call 45.7% “accuracy,” annual return, a guaranteed return, or a demonstrated large edge.
- Do not call the feature-driven warm-up completed PPO. Do not confuse its future-rank target with the next causal risk-rule target.
- Do not count duplicated histories from returned ZIPs as multiple independent runs.
- Do not claim 2023–2024 is untouched, or reuse 2025 as untouched after a test has happened.
- Do not remove baseline/initialization comparisons from scientific evidence because slide columns were simplified.
- Do not describe LoRA alone as a plug-and-play complete finance domain or synthetic router tests as real four-domain validation.
- Do not claim the knowledge base, news inputs, finance explanations, live brokerage, learned router, or flawless operation already exists.
- Do not let old README/KIT status override actual newer run records. Do not apply a historical patch in reverse to the current source snapshot.

## 14. Handover verification and scope

The supporting inventory records locally observed branches/HEADs/remotes and tracked dirty state. It scans project tree metadata to a bounded depth, excluding heavy model/data/venv trees; it is not a bytewise duplicate detector or a full source audit of every other domain. Folder timestamps establish neither exact copy dates nor completeness.

The support manifest gives copied-file hashes so the next PC can verify portability. Pretrained weights, other-domain datasets, private credentials, virtual environments and laptop caches were not duplicated; they remain on the SSD or must be recreated locally. Existing historical source trees remain intact. The main large pristine base was verified with the pinned contract; see the recorded proof.

**Next meaningful result needed:** a real inspected outcome of the prepared matched risk benchmark (or a justified revised experiment), followed by honest scientific evaluation and completion of the missing domain/runtime capabilities. The project does not need another speculative plan that ignores the completed runs.
