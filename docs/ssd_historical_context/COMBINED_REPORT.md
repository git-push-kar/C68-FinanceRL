# C68 — Three-Domain AI Systems: Unified Comprehensive Report
### Anomaly GI (TEP) · Deepfake GI (HydraFake) · Finance RL (Nifty 50 Portfolio)
#### Past · Present · Future — Problems · Solutions · Artifacts · Reproduction

> **Generated:** 2026-09-02 (rev 2026-09-02 v2 — re-audited after subtle on-disk changes) — from frozen artifacts, source code, and the three domain reports  
> **Workspace root:** `F:/`  
> **Domains on disk (re-audited 892 lines, 83k+ chars):**
> - `F:/anomaly gi/C68-Anomaly/` — `PROJECT_REPORT.md:1` (659 lines), `README.md`, `configs/config.yaml:1`, `configs/config_a5000.yaml:1`, `main.py: TEPApp`, `outputs/tep_rca_adapter/training_summary.json:1` (actual LoRA `wqkv/wo/w1/w2/w3`), `data/processed/anomaly_detector_eval.json:1`
> - `F:/deepfake gi/LY_C68/deepfake_project/` — `REPORT_exp2_to_exp3.md:1` (144 lines) + *newly audited* `RUN1/`, `RUN2/model.py:1` (CLIP+Transformer+GRU 138M), `datasets/model.py:1` (CLIP+FFT+DropPath 131M), `datasets/runs/{exp3,exp7,exp_final_single}/train.log`, `runs/intern_exp2/logs/{epochs.csv,eval_ep010.log}`, `runs/intern_exp3_corrective/logs/train.log` (5 resumes, lr 1e-5→5e-6, lm_weight 0.3→0.12)
> - `F:/C68-FinanceRL/` — `RUN_REPORT.md:1` (334 lines, **current truth**, supersedes `REPORT.md:1` 232 lines), `RUN_REPORT.md.bak`, `data/README.md:1`, `config.py:1` (`wqkv/wo`), `config.yaml:1` (stale `q_proj/v_proj`), `artifacts/finance_prod/` (7 checkpoints 10240→70144 + final `eval_2023-01-01_2024-12-31.json:1` `141,251/-6.13%`), `requirements.txt`
> **Shared hardware target:** NVIDIA RTX A5000 (24 GB, Ampere sm_86, CUDA 12.1/13.2, driver 595.95) — all three projects target the same GPU  
> **No code was re-executed to generate this report** — all numbers are quoted from existing artifacts and reports; subtle on-disk deltas since last revision are now incorporated (see §7.5).

---

## Table of Contents
1. [Executive Summary — One Page per Domain](#1-executive-summary)
2. [At-a-Glance Comparison](#2-at-a-glance-comparison)
3. [PART I — Anomaly GI: TEP Industrial Anomaly Detection & Auto-RCA](#3-part-i--anomaly-gi)
   - 3.1 Objectives & Design Principles
   - 3.2 Dataset (Rieth Consolidated)
   - 3.3 System Architecture (Two-Brain)
   - 3.4 Current Implementation (Preprocessing → Detector → Evidence → Streaming → LLM)
   - 3.5 Progress Report & Timeline
   - 3.6 Results — Window / Event / Sensor / LLM
   - 3.7 The Fault 15 Story (5-Stage Diagnosis)
   - 3.8 All 20 Faults — Stealth Class 3/9/15
   - 3.9 What We Tried (Negative Results)
   - 3.10 Lessons, Pitfalls, Future Scope
4. [PART II — Deepfake GI: HydraFake + InternVL3-2B Reasoning](#4-part-ii--deepfake-gi)
   - 4.1 Objectives & Setup
   - 4.2 Dataset (HydraFake) & Model
   - 4.3 Training History (exp1 → exp2 → exp3_corrective)
   - 4.4 Eleven Problems Found (P1–P11)
   - 4.5 Corrective Changes (dataset.py / train.py / model.py / inference.py)
   - 4.6 Sequence Length 192→512 Study
   - 4.7 Results & Artifacts
   - 4.8 Future Scope
5. [PART III — Finance RL: InternLM2.5 + PPO Portfolio Management](#5-part-iii--finance-rl)
   - 5.1 Objectives & Plain-Language Model
   - 5.2 Dataset (Nifty 50 Panel) & Environment
   - 5.3 Model & PPO Implementation
   - 5.4 Hardware & Performance (A5000 Verified)
   - 5.5 Progress, Training Evidence & Honest vs Leaky
   - 5.6 Limitations & Future Scope
6. [PART IV — Cross-Cutting Synthesis](#6-part-iv--cross-cutting-synthesis)
7. [PART V — Artifacts Inventory (Tables, Graphs, Checkpoints)](#7-artifacts-inventory)
8. [PART VI — How to Reproduce Everything](#8-how-to-reproduce)
9. [Appendices — File Maps & Configs](#9-appendices)

---

## 1. Executive Summary

### Anomaly GI — TEP (`anomaly gi/C68-Anomaly/PROJECT_REPORT.md:4`)
> One-line: **52-sensor chemical plant watcher** — LSTM autoencoder learns "normal" unsupervised, flags anomalies via reconstruction error, packages evidence into a forensic JSON, and a **frozen InternVL2-2B + one LoRA `tep_rca`** explains *what/why/severity/action* in auditable English. Detector is perfect on 17/20 faults; the scientific contribution is characterizing **why faults 3, 9, 15 are stealth** (weak, distributed, smooth, relationship-preserving, reconstructible) and proving three plausible fixes fail.

**Status Sep 2026:** Detector pipeline complete, rigorously evaluated (10,075 runs, 897k windows). Adapter `tep_rca` trained (1008/40/40 split by fault, 62 MB) and evaluated (subsystem 1.00, hallucination 0.00, exact-ID 0.50). Streaming `TEPApp` + FastAPI + Streamlit + SQLite live.

### Deepfake GI — HydraFake (`deepfake gi/LY_C68/deepfake_project/REPORT_exp2_to_exp3.md:5`)
> One-line: **InternVL3-2B (InternViT 448px + Qwen2 + LoRA r64)** trained to output **structured forensic reasoning** `<fast><planning><reasoning><reflection><conclusion><answer>` + binary real/fake classifier. `exp2` ran 11 epochs on HydraFake, then **reasoning degraded** due to 11 bugs (mixed targets, no EOS, duplicate loading, BF16 crash, stochastic decoding, scheduler patchwork…). **Corrective run `intern_exp3_corrective`** freezes vision+classifier+projector, retrains only LoRA on a single coherent tagged+EOS format, raises `lm_seq_len 192→512` to avoid truncating `</answer>`.

**Status Aug 2025+:** Trusted checkpoint `runs/intern_exp2/checkpoints/ep010.pth` (5.16 GB ZIP_OK, VaAUC 0.943) retained; `ep011` and `best.pth` discarded; corrective 1 epoch done (`epochs.csv` joint loss 0.712/0.892), greedy deterministic decoding enabled.

### Finance RL — Nifty 50 (`C68-FinanceRL/REPORT.md:1`)
> One-line: **Frozen InternLM2.5-1.8B + LoRA `finance` (wqkv/wo r16) + 3 MLP encoders + Actor/Critic** learns **long-only, cost-aware portfolio weights** via PPO in a simulated market (30-day window ×49 assets, 10 bps cost, 30% cap, reward = net − turnover/vol/drawdown penalties). **Bundled panel `data/nifty50_prices.csv` 287k rows, 49 tickers, 2017→2026** (survivorship-biased, acknowledged).

**Status Sep 2026:** Pipeline stable on A5000 (bf16, 3.9 GB, 1.3h/75k steps). First honest result: **512-step honest smoke beats equal-weight by +0.47% on untouched 2023-24** (151,186 vs 150,471, Sharpe 1.82 vs 1.93, turnover 0.099→0.002 as training progresses). Leaky 25k run +4.43% shows honesty gap to be closed by 75–100k honest production run.

---

## 2. At-a-Glance Comparison

| Dimension | Anomaly GI (TEP) | Deepfake GI (HydraFake) | Finance RL (Nifty 50) |
|-----------|------------------|-------------------------|-----------------------|
| **Goal** | Detect unknown faults + explain RCA | Classify real/fake + generate forensic reasoning trace | Learn portfolio allocation that grows wealth risk-adjusted |
| **Data** | Rieth TEP: 52 sensors, 250k normal + 5M faulty rows, fault at sample 160 (`configs/config.yaml:39`) | HydraFake: 48,320 unique images, 144k JSON entries (11% tagged), 4 test splits ID/CM/CF/CD | Nifty 50 long panel 287,263 rows (`data/nifty50_prices.csv`), 2,027 trading days (`env.py:45`) |
| **Backbone** | LSTM-AE (128h×2, latent 64) + InternVL2-2B (InternLM2-Chat-1.8B) | InternVL3-2B (InternViT + Qwen2) | InternLM2.5-1.8B-chat (frozen) |
| **Adapter** | `tep_rca` LoRA r16 α32 on `q/k/v/o gate/up/down` (`configs/config.yaml:156`) | LoRA r64 α128 on `q/k/v/o gate/up/down` (`model.py:172`) | `finance` LoRA r16 α32 on `wqkv/wo` (`config.py:14`) |
| **Training principle** | Unsupervised detector (normal-only) + supervised LLM (fault-disjoint splits) | cls_only 3ep → joint 2–8ep; corrective freezes cls/projector | PPO γ0.99 λ0.95 clip 0.2, frozen LLM + trainable encoders/heads |
| **Key metric** | Event rate 1.00 on 17/20, 0.118–0.196 on stealth 3/9/15; FPR 1.42% win / 9.3% event | Val AUC 0.943 (ep010), train acc 0.834 corrective | Honest holdout 2023-24: +0.47% alpha (512 steps), Sharpe 1.82 |
| **Central failure** | Stealth drifts indistinguishable at 60×52 global-MSE | Mixed targets + no EOS + duplication + BF16 crash | Needs 75–100k steps to stabilize; survivorship bias |
| **Hardware** | A5000 24GB, bf16 QLoRA 6-8GB or BF16 12-14GB | A5000 bf16, 2.77s/step @512 (vs 0.76@192) | A5000 bf16 3.9GB, 1.3h/75k, TF32, WDDM |
| **Docs** | `PROJECT_REPORT.md` 659 lines + `README.md` 304 lines | `REPORT_exp2_to_exp3.md` 144 lines + in-code fix comments | `REPORT.md` 232 lines + `README.md` 94 lines |

```
Domains share:  Frozen LLM + small LoRA + honest splits + A5000 bf16 + P1-pitfall honesty
   TEP:  normal-only autoencoder — "healthy sound" prior
   Deepfake:  richest-tag wins — "one voice" prior
   Finance:  equal-weight baseline — "do nothing smart" prior
```

---

## 3. PART I — Anomaly GI

> Full source: `F:/anomaly gi/C68-Anomaly/PROJECT_REPORT.md:1` — this section is a faithful expansion with all tables and figures preserved.

### 3.1 Objectives — What We Set Out to Build and Why

**Real-world problem (`PROJECT_REPORT.md:34`):** Chemical plants stream 52 sensors every few seconds. Per-sensor thresholds alarm but don't explain *why* or *what to do*. Late detection = off-spec, damage, safety risk.

**Two-brain design (`PROJECT_REPORT.md:38`):**

| Brain | Job | Analogy |
|-------|-----|---------|
| **Sensor brain (unsupervised)** | Learn normal from *only* normal; flag reconstruction failures. No fault labels. | Veteran operator who knows healthy sound |
| **Language brain (supervised adapter)** | Evidence JSON → grounded RCA report + follow-up QA | Senior engineer writing incident report |

**Design principles (`PROJECT_REPORT.md:45`):**
1. No fault labels for detection — must work on unknown future faults.
2. Explainability by construction — every claim traceable to deviation % / z / trend / onset order; disclaimer "correlation ≠ causation".
3. Separation of concerns — `tep_rca` LoRA holds *only* language reasoning; scaler/AE/threshold/evidence outside → swappable.
4. Production streaming — `TEPApp.process_sensor_stream` one record at a time, 3-consecutive confirmation, SQLite `events/event_store.py`, FastAPI `api/server.py` + Streamlit `ui/app.py`.

**End-to-end journey (`PROJECT_REPORT.md:52`):**
```
sensor stream (52) → 60-window stride5 → scaler → LSTM-AE MSE → threshold 0.687
  → 3-consecutive aggregation → build_event (top5, deviation%, z, trend, onset order)
  → InternVL2-2B+tep_rca → JSON {summary,root_cause,subsystem,evidence,reasoning,severity,confidence,action,uncertainty}
  → SQLite → "Should I check condenser valve?" → grounded answer
```

### 3.2 Dataset — What Data We Have

**Raw layout (`PROJECT_REPORT.md:67`, `preprocessing/tep_loader.py`):** Rieth et al. consolidated, auto-detected (legacy 52-col also supported):

| File | Rows | Content |
|------|------|---------|
| `data/raw/normal/TEP_FaultFree_Training.csv` | 250,000 (500×500) | fault 0 |
| `data/raw/normal/TEP_FaultFree_Testing.csv` | 480,000 (500×960) | fault 0 |
| `data/raw/faults/TEP_Faulty_Training.csv` | 5,000,000 (20×500×500) | faults 1–20 |
| `data/raw/faults/TEP_Faulty_Testing.csv` | 9,600,000 (20×500×960) | faults 1–20 |
| Columns | `faultNumber, simulationRun, sample, xmeas_1..41, xmv_1..11` | 52 sensors |
| Fault injection | sample 160 (0-based, 161 1-based) | `configs/config.yaml:39` |

Sensor vocab 52: 41 XMEAS + 11 XMV; names/subsystems in `evidence/process_relationships.py` (e.g., `A_Feed_Flow→feed_system`, `Condenser_Cooling_Water_Flow→condenser_cooling_system`) (`PROJECT_REPORT.md:79`).

**Processed after `scripts/prepare_tep.py` (`PROJECT_REPORT.md:82`):**

| Artifact | Location | Size / Count |
|----------|----------|--------------|
| Scaler (fitted ONLY on normal training) + baseline stats | `outputs/preprocessing/scaler.pkl`, `baseline_stats.json` | 52 features |
| Leakage-free windows (by whole simulationRun) | `data/processed/normal_windows_{train,val,test}.npy` | 44,500 windows [60×52], split 31,150/6,675/6,675; block 500, test 0.2, val 0.15 |
| Per-fault arrays | `data/processed/fault_values/fault_01..20.npy` | 730k samples each (Training+Testing merged) |
| Manifests | `data/processed/manifests/detector_split.json` | 350/75/75 runs, seed 42 |

> Invariant: scaler/baseline from `TEP_FaultFree_Training.csv` only — no fault, no Testing normal leakage (`PROJECT_REPORT.md:90`).

**LLM supervision (`PROJECT_REPORT.md:92`):**
- Truth: `scripts/fault_knowledge.py` — 22 fault scenarios (initiating sensor, cascading delays, severity, reasoning, action). 16–20 labeled `unknown`.
- Generation: `scripts/generate_llm_dataset.py` + `generate_synthetic_rca_v2.py` → `evidence JSON → RCA JSON` pairs. Splits **by fault ID**: train {1,2,3,5,6,7,8,9,10,11,12,13,16,17,18,19,20,22} / val {4,14} / test {15,21} → `data/llm/split_metadata.json` 1008/40/40; also `data/processed/rca_real/`.
- v2 calibrated: `outputs/llm_dataset_v2/synthetic_rca.jsonl` 400 ex (20/fault) sampled from real z-distributions.

### 3.3 System Architecture at a Glance

```
┌──────────────────────────────────────────────────────────────────────┐
│ Layer A — Sensor / Time-Series (no LLM)                              │
│ preprocessing/ → anomaly_detection/ → evidence/ → events/             │
│ streaming/ (simulator)                                               │
│ Layer B — Language (frozen LLM + one adapter)                        │
│ llm/ (InternVL2-2B + tep_rca LoRA/QLoRA)                              │
│ Orchestration: main.py: TEPApp                                       │
│ Interfaces: api/server.py (FastAPI) + ui/app.py (Streamlit)          │
└──────────────────────────────────────────────────────────────────────┘
Data flow:
  sensor_record {values:[52], sample_index}
    → deque 60 → stride5 → scaler → LSTM-AE (global MSE + per-sensor)
    → threshold 0.687 → recent_flags → _feed_aggregator
        (3 consec anomalous → open; 20 normal → close; cap 200)
    → build_event() → EventEvidence {score, top5 (dev%,z,trend), temporal_sequence, subsystem, disclaimer}
    → RCAInference.generate_report() OR deterministic fallback → JSON
    → EventStore SQLite outputs/events.db → answer_followup()
```

Why two layers (`PROJECT_REPORT.md:126`): detector works offline; LLM never sees raw sensors, only evidence JSON → enforced "does not prove causation".

### 3.4 Current Implementation — How Each Part Works Today

#### 3.4.1 Preprocessing (`preprocessing/`)
- `tep_loader.py`: auto-detect Rieth vs legacy, groups by `faultNumber/simulationRun/sample`, validates row counts, interpolates/ffill/drops per `dataset.missing_value_strategy`, streams per-run (never 5GB at once).
- `scaler.py`: `StandardScaler` fitted only on normal training; persists `scaler.pkl` + `BaselineStats`; `load_scaler()` sole source for z-scores.
- `windowing.py`: `to_windows(arr,60,5)` → `[N,60,52]` never across `simulationRun`; leak-free by `block_samples=500`; `float32`.

#### 3.4.2 Anomaly Detector (`anomaly_detection/`)
- **Model `anomaly_detection/lstm_autoencoder.py`:** Encoder LSTM(52→128×2) → latent 64; Decoder latent→h0/c0 + repeated latent → LSTM(64→128×2) → linear 52.
- **Training:** *Only* normal windows, MSE, AdamW lr1e-3 cosine, 100ep batch64 (A5000 128), early-stop patience15, gradclip1.0, ckpt every5 → `outputs/anomaly_detector/model.pt` (2 MB).
- **Threshold:** normal val reconstruction errors, `percentile p=99.0` (A5000 p99.5), frozen **0.687363** (mean0.619 std0.028 n6675) — `outputs/anomaly_detector/threshold.json`.
- **Inference `anomaly_detection/inference.py`:** `score_window` → scalar `mean(per-sensor MSE)` + vector[52]; `is_anomalous(score>thr)`; `score_windows` batched.
- Legacy: `lstm_predictor.py` + `train_predictor.py` (evaluated, not adopted).

#### 3.4.3 Evidence Extraction (`evidence/`)
- `sensor_contribution.py`: per-sensor error→contrib%, deviation% `(current-baseline)/baseline*100`, z, ranking; `event_level_deviations()` aggregates with `min_deviation_percent 3.0`.
- `temporal_analysis.py`: `detect_onsets()` (>1.5×std), `analyze_temporal_sequence()` (sorted onsets→minutes), `sensor_trend()` (linear fit window8→increasing/decreasing/stable), `pre_post_context()`.
- `process_relationships.py`: 52-name vocab, subsystem groups, `KNOWN_PROCESS_RELATIONSHIPS` (e.g., Cooling Water→Reactor Temp→Pressure), `suggest_subsystems()` heuristic, always `MODEL_DERIVED` + disclaimer.
- `event_builder.py`: assembler → `AnomalyEvent` + `EventEvidence`; severity from `score/thr`: ≥3.0 critical, ≥1.8 high, ≥1.2 medium, else low; includes temporal_sequence up to 6, top5, candidate_subsystem, reasoning_notes ("does not prove causation").

#### 3.4.4 Streaming & Event Aggregation (`streaming/`, `events/`)
- `streaming/simulator.py`: `SensorStream` replays CSVs live (`window/stride/replay_rate`); `TEPApp.run_stream_from_file()` can splice fault at `inject_fault_at` (default 160).
- `main.py: TEPApp`: public API `process_sensor_stream(record)`, `finalize_stream()`, `answer_followup()`, `run_stream_from_file()`; state deque60, `consecutive_windows_to_confirm=3`, `min_separation=20`, `max_event=200`.
- `events/event_store.py`: SQLite `outputs/events.db` — events + reports + conversation history.

#### 3.4.5 LLM Adapter (`llm/`)
- **Base:** `OpenGVLab/InternVL2-2B` (InternLM2-Chat-1.8B), `trust_remote_code=True`, text-only dummy `pixel_values` + `image_flags=0`.
- **Adapter:** single LoRA/QLoRA `tep_rca` (`r16 α32 dropout0.05`). **Subtle:** `configs/config.yaml:156` *declares* targets `q_proj/k_proj/v_proj/o_proj gate/up/down` (InternLM2 legacy doc), but the **actually trained** adapter in `outputs/tep_rca_adapter/training_summary.json:9` is `["wqkv","wo","w1","w2","w3"]` — the correct InternLM2 fused names (see Finance fix `config.py:14`). The on-disk artifact is authoritative; YAML is stale doc. A5000 QLoRA 4-bit vs BF16 variant stays `w1/w2/w3` FFN included. → `outputs/tep_rca_adapter/tep_rca/` (~62 MB).
- **Training:** `llm/train_adapter.py` / `scripts/train_tep_adapter.py`; QLoRA 4-bit NF4 double-quant bf16 ~6-8GB or BF16 LoRA ~12-14GB on A5000; batch2×8 (A5000 BF16 4×4), 3ep lr2e-4 warmup0.03 cosine; splits 1008/40/40; saves `training_summary.json` (`outputs/tep_rca_adapter/training_summary.json`).
- **Inference:** `llm/inference.py` → `RCAInference.generate_report(evidence)` + `answer_followup(event,report,history,q)`; deterministic fallback `main.py:_fallback_report()` when adapter absent.
- **Evaluation:** `llm/evaluate.py` + `scripts/benchmark_tep_adapter.py`.

#### 3.4.6 Config & Utils
- Single source `configs/config.yaml:1` merged over `utils.py:_DEFAULT_CONFIG`; every script `--config`; `utils.py` provides `load_config()`, `load_json/save_json`, `get_device()`, `set_seed(42)`, `gpu_memory_summary()`.

### 3.5 Progress Report — What Is Done vs Pending

**Timeline condensed (`PROJECT_REPORT.md:181`):**

| Phase | What happened | Outcome |
|-------|---------------|---------|
| 0 Scaffolding | Repo `preprocessing/ anomaly_detection/ evidence/ llm/ streaming/ events/ api/ ui/ configs/ scripts/ main.py` | 300-line README |
| 1 Preprocessing | `tep_loader+scaler+windowing`, `prepare_tep.py --full` Rieth | 44.5k normal windows, scaler locked, leak-free |
| 2 Detector training | LSTM-AE 128/64 dp0.05 on 31k windows 50→100ep, threshold 99→99.5 | model 2MB, thr0.687, FPR1.4% prec0.999 |
| 3 Detector eval | `evaluate_anomaly_detector.py` on held-out + all faults | F1 0.861 AUROC 0.886 delay6.25 but 3 faults miss |
| 4 LLM dataset | `fault_knowledge.py` 22 faults + `generate_llm_dataset.py` + `generate_synthetic_rca_v2.py` 400 | Hf + synthetic |
| 5 Adapter training | `tep_rca` InternVL2-2B QLoRA/bf16 3ep 1008 ex | Adapter62MB eval1.0 ckpts135/189 |
| 6 Fault15 deep-dive | 5-stage suite global/per-sensor/z/temporal/persistence/corr | Diagnosed stealth |
| 7 All-fault sweep | `evaluate_all_faults_full.py` 20×500 runs | Ranked 20; 3,9,15 problematic 11–19% |
| 8 Comparative | `diagnose_faults_3_9_15.py` 52×52 corr gradients | Proved common mode |
| 9 Alternative detectors | top-k, prediction, relationship | Documented negative |
| 10 E2E | `test_end_to_end.py` + `test_adapter.py` + API/UI | Pipeline live |

**Checklist (`PROJECT_REPORT.md:197`):**

| Component | Status | Evidence |
|-----------|--------|----------|
| Raw→processed pipeline | **Done** | `prepare_summary.json`, 44.5k windows |
| Scaler/baseline (normal-only) | **Done** | `scaler.pkl`, `baseline_stats.json` |
| LSTM-AE + threshold | **Done** | `model.pt` 2MB, `threshold.json`0.687 |
| Window-level eval | **Done** | `anomaly_detector_eval.json` |
| Event aggregation (3/20/200) | **Done** | `main.py:_feed_aggregator` + `events.db` |
| Evidence builder | **Done** | `evidence/event_builder.py` |
| Fault knowledge base | **Done** | `fault_knowledge.py` 22 entries |
| LLM dataset | **Done** | `data/llm/{train,val,test}`, `outputs/llm_dataset_v2/`400 |
| Adapter training | **Done** | `outputs/tep_rca_adapter/tep_rca/` |
| Adapter eval | **Done** | `evaluation.json` root-cause0.50 |
| Fault15 diagnosis | **Done** | `outputs/evaluation/fault15_*` 7 files |
| All-fault eval | **Done** | `all_faults_detector_*` per-run CSV + summary |
| 3/9/15 comparative | **Done** | `faults_3_9_15_final_report.md` |
| Alternative detectors | **Done (negative)** | `prediction_vs_reconstruction_*`, `relationship_*` |
| Streaming simulator + API/UI | **Done (smoke)** | `streaming/simulator.py`, `api/server.py`, `ui/app.py` |
| **Comprehensive threshold sweep** | **Not done** | `tmp_threshold_sweep.py` exists, not run systematically |
| **Broader hyperparam search** | **Pending** | Single config 60/5 128/64 only |
| **Production hardening** | **Pending** | Basic FastAPI |
| **Human expert validation** | **Pending** | Automated metrics only |

### 3.6 Results — How Well Does It Work?

#### 6.1 Window-level (held-out, `anomaly_detector_eval.json`) — threshold 0.687 on 6,675 normal val; tested on 6,675 normal test + 2,919,780 fault windows (`PROJECT_REPORT.md:225`):

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Threshold | 0.687 (mean0.619 std0.028 p99) | >0.687 = anomalous |
| Precision | 0.99996 (95FP/2,208,046TP) | When fires, almost always real |
| Recall (TPR) | 0.756 (2,208,046/2,919,140) | 76% fault windows |
| F1 | 0.861 | Balance |
| FPR | 1.42% (95/6,675) | ~1/70 normal windows alarms |
| FNR | 24.4% (711k missed) | Dominated by stealth |
| AUROC | 0.886 | Good separator |
| AUPRC | 0.9997 | Excellent imbalanced |
| Mean delay | 6.25 samples | Worst 35 |

Per-fault delay: **0** = faults 1,2,4,5,6,7,8,10,11,12,13,14,16,19,20 instant; **5**=18; **15**=17; **35**=3,9,15 slow/late (**problem trio**).

```
Detector performance spectrum (PROJECT_REPORT.md:245):
  17 faults at 89/89 windows (scores 0.85–632)  vs  3 faults at 1.1–1.8/89
```

#### 6.2 Event-level (all-fault sweep, frozen thr0.687, event=≥3 consec) (`PROJECT_REPORT.md:248`):

| Fault | Runs | Event rate | Mean max | Mean # above | Median delay | Verdict |
|-------|------|------------|----------|--------------|--------------|---------|
| Normal |75|0.093 (7/75 false)|0.683|1.28|—|Baseline|
| 1 A/C feed ratio step |500|1.000|34.96|89.0|0.0|Easy|
| 2 B composition step |500|1.000|60.48|89.0|0.0|Easy|
| **3 D feed temp step**|500|**0.118**|**0.678**|1.07|41.3|**Hard**|
| 4 Reactor CW temp step|500|1.000|1.587|89.0|0.0|Easy|
| 5 Condenser CW temp|500|1.000|3.966|88.9|0.0|Easy|
| 6 A feed loss|500|1.000|632.7|89.0|0.0|Easy|
| 7 C header pressure|500|1.000|25.58|89.0|0.0|Easy|
| 8 A,B,C random|500|1.000|27.05|88.9|0.0|Easy|
| **9 D feed temp random**|500|**0.126**|**0.679**|1.14|40.3|**Hard**|
| 10 C feed temp random|500|1.000|0.927|66.9|6.6|OK|
| 11 Reactor CW random|500|1.000|2.303|89.0|0.0|Easy|
| 12 Condenser CW random|500|1.000|77.40|89.0|0.0|Easy|
| 13 Reaction kinetics drift|500|1.000|47.92|86.4|2.6|Easy|
| 14 Reactor CW valve sticking|500|1.000|10.93|89.0|0.0|Easy|
| **15 Condenser CW valve sticking**|500|**0.196**|**0.684**|1.82|39.2|**Hard**|
| 16 unknown|500|1.000|0.856|67.5|4.3|OK|
| 17 unknown|500|1.000|57.17|87.2|1.8|Easy|
| 18 unknown|500|1.000|603.39|83.3|5.7|Easy|
| 19 unknown|500|1.000|0.887|88.8|0.2|Easy|
| 20 unknown|500|1.000|2.078|85.4|3.6|Easy|

> **Takeaway:** 17/20 at 100% with scores 0.85–632; 3/9/15 at 11–19% with scores 0.678–0.684 **indistinguishable from normal 0.683** (`PROJECT_REPORT.md:276`).

Visual — Event rate vs Mean max (`PROJECT_REPORT.md:278`):

```
Event rate 1.0 ┤ ●●●●●●●●●●●●●●●●●  (17 easy)
             │                   ╲
             │                    ╲
           0.5┤                     ╲
             │                      ╲
           0.2┤                       ● 15(0.196,0.684)
           0.1┤                  ● 9(0.126)  ● 3(0.118)
             └────────────────────────────────────────
              0.67 0.68 1.0  2.0  10  50 100 632 mean_max
                             ▲ normal 0.683 beside 3/9/15
```

#### 6.3 Sensor-level forensics Fault15 (`fault15_sensor_reconstruction.csv`):

| Ranking | Top sensor | Normal mean | Fault15 mean | Ratio | Runs >p95 |
|---------|------------|-------------|--------------|-------|-----------|
| 1 | XMEAS_11 |0.338|0.410|**1.21**|17/500 (3.4%)|
| 2 | XMEAS_22 |0.776|0.914|1.18|14/500 (2.8%)|
| 3 | XMEAS_5 |0.999|0.999|1.00|0/500|

Best ratio 1.21, only 3.4% elevated; Top3 vs Global `global0.684 top1 0.91 top3 0.59 top5 0.59` — top3 < global → dilution hypothesis rejected (`PROJECT_REPORT.md:303`).

#### 6.4 LLM Adapter (`evaluation.json` + `evaluation_large_20perFault.json`, 40 samples 20×15+20×21):

| Metric | Value | Meaning |
|--------|-------|---------|
| Fault classification | **1.00** | Subsystem always right |
| Evidence consistency | **1.00** | Never contradicts evidence |
| JSON validity | **1.00** | Always valid |
| Hallucination | **0.00** | Never invents sensors |
| Hedged reasoning | **1.00** | Always "correlation ≠ causation" |
| Recommendation | **1.00** | Always actionable |
| Root cause exact ID | **0.50** | Half exact name (see nuance) |
| Severity | **0.325** | Weak grading |

Nuance 0.50 (`PROJECT_REPORT.md:318`): all 20 Fault15 labeled as Fault12 (both condenser, closely related); Fault21 hedged between A feed loss vs A/C ratio. Subsystem 1.00 is operationally correct; 0.00 overclaiming. Training `tep_rca` r16 α32, 1008 train, 40val (4&14), 40test (15&21), 3ep, ckpts 135&189 (`outputs/tep_rca_adapter/training_summary.json`).

### 3.7 The Fault 15 Story — How a Single Fault Exposed Systemic Problem

**How found (`PROJECT_REPORT.md:330`):** Overall recall 0.756 hid per-fault breakdown: Fault15 mean max 0.684 vs normal 0.683 — 0.3% diff.

**5-stage frozen-detector suite** (75 normal + 500 Fault15, 89 windows each, thr0.687) (`PROJECT_REPORT.md:337`):

| Stage | Question | Measured | Result Fault15 |
|-------|----------|----------|----------------|
| 1 Global | Above threshold? | max/mean/p95/p99/n_above/max_consec/first | **No.** Normal 0.682±0.018 vs Fault15 0.684±0.019 overlap 99.7%; only 40.6% ≥1 above; median n_above 0 |
| 2 Per-sensor | Single sensor diluted? | Per-sensor MSE 52, top-k vs global, 5 rankings | **No dilution.** Best ratio1.21 3.4% elevated; Top3 0.59<global0.62 opposite |
| 3 Physical z | Raw deviating? | \|z\| per window, crossings ≥2/3/4 | **Weak.** Top post_mean \|z\| XMEAS_22 0.87 (+8% vs 0.80), max1.38; 500/500 cross ≥2 (noise), ≥3 never |
| 4 Temporal | Clear onset order? | First crossing sample at \|z\|≥2/3/4 | **No order.** Scattered/random; no chain like Fault1 |
| 5 Persistence | Accumulates? | n_above at 0.60/0.62/0.64/0.66/0.687, cumulative excess | **Weak.** At0.687 1.28→1.81 (+0.5), max_consec1.01→1.37, never 3 needed; excess 5.3→6.2 (+15%) |
| 6 Correlation | \|z\|→error? | Corr \|z\| vs per-sensor error post-onset top10 | **Only 1/10.** XMEAS_22 0.70, others 0.02–0.25 — AE reconstructs 9/10 well |

Outputs: `fault15_score_distribution.csv`, `fault15_diagnostic_global.csv` (575 runs), `fault15_sensor_reconstruction.csv` (52 rows), `fault15_global_vs_topk.csv` (5.6MB), `fault15_sensor_run_statistics.csv` (2.9MB), `fault15_sensor_zscores.csv`, `fault15_zscore_crossings.csv` (2.5MB), `fault15_persistence.csv`, `fault15_temporal_onset.csv`, `fault15_zscore_vs_reconstruction.csv`.

**Diagnosis ranked causes (`PROJECT_REPORT.md:349`):**
- Primary 70%: (E) Genuinely too close to normal (+8% physical, +0.3% recon) — AE has seen similar normal variations.
- Secondary 20%: (D) AE reconstructs too well — corr 0.02–0.25 for 9/10 sensors, latent preserves fault trajectory.
- Minor 10%: (C) Low-amplitude persistence — slightly more windows above but not 3-consecutive.
- Rejected: (A) Dilution, (B) Temporal blindness.

Bottom line: Not a bug — class of fault any 60-window 52-sensor reconstruction method finds hard: slow, distributed, smooth drift inside normal noise.

### 3.8 Testing All 20 Faults — Who Is Problematic and Why

**Full sweep `scripts/evaluate_all_faults_full.py` — frozen detector thr0.687, 75 normal + 20×500=10,000 fault runs (10,075; 89 windows/run → ~897k windows)** (`PROJECT_REPORT.md:365`):

| Group | Faults | Event rate | Mean max | Why |
|-------|--------|------------|----------|-----|
| Easy (17) |1,2,4,5,6,7,8,10,11,12,13,14,16,17,18,19,20|**1.000**|0.86–632|Strong localized high deviation; AE cannot recon |
| Hard (3) |**3,9,15**|**0.118,0.126,0.196**|0.678,0.679,0.684|Weak distributed smooth relationship-preserving reconstructible|
| Control |Normal|0.093|0.683|—|

Rankings: Easiest 1,2,4; hardest 15>9>3; most overlap 15,9,3; strongest persistence 1,2,4 vs weakest 15,9,3.

Per-fault details from `all_faults_detector_summary.csv` (`PROJECT_REPORT.md:386`):
```
fault window_rate event_rate mean_max median_max mean_n_above max_consec delay
  3    0.294       0.118     0.678   0.677     1.07      0.84    41.3
  9    0.298       0.126     0.679   0.678     1.14      0.92    40.3
 15    0.406       0.196     0.684   0.683     1.82      1.37    39.2
 normal  —         0.093     0.683   0.680     1.28      0.99     —
  6     1.0        1.000   632.7   570.6      89.0      89.0     0.0 ← contrast
```

**Comparative deep-dive 3/9/15 vs Normal (`diagnose_faults_3_9_15.py`, 1575 runs)** (`PROJECT_REPORT.md:397`):

| Metric | Normal | Fault3 | Fault9 | Fault15 | Interpretation |
|--------|--------|--------|--------|---------|----------------|
| Grad mean |1.706±0.04|1.703±0.03|1.703±0.03|1.709±0.03|Identical — smooth drifts|
| Corr change pre→post 52×52|0.108±0.01|0.107±0.01|0.108±0.01|0.108±0.01|Identical — preserved|
| z_post_max|0.30±0.11|0.34±0.09|0.31±0.11|0.31±0.11|Weak <0.5σ|
| Score max|0.682±0.018|0.678±0.019|0.679±0.019|0.684±0.019|Identical/lower — reconstructs better|
| n_above|1.28|1.07|1.14|1.82|Slightly higher 15|
| Event rate|0.093|0.118|0.126|0.196|All low|

Conclusion: **Five signatures of stealth** — weak (<0.5σ), distributed, smooth, relationship-preserving, AE-reconstructible — fail for same reason (`PROJECT_REPORT.md:408`).

### 3.9 What We Tried to Fix It — Experiments and Why They Did (or Didn't) Help

> Every experiment frozen detector, no retraining, no test-fault tuning (`PROJECT_REPORT.md:414`).

| Approach | Effect Fault15 | Effect Normal FPR | Verdict |
|----------|----------------|-------------------|---------|
| **Lower global threshold** (e.g., 0.65) | marginal | **60% at 0.60** (3/3 normal false) → explode | Reject |
| **Sensor-aware top-k** (`mean(top3)`) | No gain ratio1.21 | FPR 28% worse than global p99 0.68 (30%/38%) | Reject (`diagnose_sensor_aware.py` top1 p99 1.86 → FPR28% TPR24%) |
| **Prediction-based** (predict next window LSTM) `lstm_predictor.py` (`PROJECT_REPORT.md:429` table) | Worse: Recon 0.684→Pred 0.983; Fault lower than normal 1.003 → event 0.00 | Good 0% but no detection | Reject |
| **Relationship-violation** (9 types act/lag/pair) `relationship_detector_summary.csv` | Delta 0.01–0.06 (e.g., A_C_Feed→Reactor_Feed 0.962→0.960, Condenser 1.233→1.297) | Neutral | Reject |
| Per-fault thresholds | — | — | Reject violates population eval |
| **Prediction+recon hybrid / Longer window 500 / Multi-scale voting / Constrained AE / SVDD** | Not yet tried | — | **Proposed** |

Prediction vs Recon table (`PROJECT_REPORT.md:430`):

| Condition | Recon mean_max | Recon event | Pred mean_max | Pred event |
|-----------|---------------|-------------|---------------|------------|
| Normal |0.683|0.093|1.003|**0.00**|
| Fault3 |0.678|0.118|0.979|**0.00**|
| Fault9 |0.679|0.126|0.982|**0.00**|
| Fault15|0.684|0.196|0.983|**0.00**|

Negative results valuable: not dilution, not temporal, not relational — genuine indistinguishability at 60/52/global-MSE.

### 3.10 Key Observations, Faulty Errors, Lessons Learned

**What does well (`PROJECT_REPORT.md:475`):**
- Near-perfect 17/20, FPR1.42% win /9.3% event precision0.99996, leak-free, LLM 100% subsystem/0% hallucination, streaming-ready.

**Central failure stealth:** faults 3,9,15 — 1,500 runs 80–88% miss.

**Faulty errors & pitfalls (`PROJECT_REPORT.md:488`):**
1. Threshold p99.0 vs p99.5 tug-of-war (FPR13.5%→<5% but pushes stealth below).
2. Normal \|z\| ~0.80 itself → fault 0.87 only +8% margin.
3. Top-k illusion — verify with data before building scorer.
4. LLM confusion 15 vs 12 (both condenser) → severity 0.325 noise.
5. 500-sample vs 60-window mismatch (12% trajectory visible; slow drift invisible).

**Lessons:** Reconstruction≠anomaly (capacity must be constrained); Global MSE strong default but not universal; Fault taxonomy by dynamics not ID; Negative results deserve reports.

### Future Scope (`PROJECT_REPORT.md:505`)

**Near-term (no retraining, new scoring heads on frozen detector):**

| Proposal | Rationale | Test | Risk |
|----------|-----------|------|------|
| Temporal prediction head (longer horizon 5-ahead or delta) | Stealth predictable locally but err cumulatively (+15% excess) | Train on normal only, p99 on 75 normal val | Low |
| Multi-scale windows 30/60/120/500 voting | 60 smooths drift; 500 sees mean shift | OR anomaly if any branch fires | Medium |
| Per-sensor z-threshold bypass (calibrated on \|z\|>1.5) | XMEAS_22 corr0.70 informative; prev top-k wrong calibration | Threshold on z directly | Low |
| Threshold sweep ROC 0.55–0.80 | Fixed 0.687 may not optimal | TPR@FPR 1/5/10% | Low |

**Medium-term (retrain normal-only):** Constrained AE (latent16→8), One-class Deep SVDD/USAD, Self-supervised contrastive.

**LLM:** Expand 400→500–600 (20–30/fault) + real `detector_derived_rca.jsonl`, calibrate severity, retrieval grounding citations, lightweight 20-way classifier head for exact-ID, human expert validation.

**Product:** auth/rate-limit/monitoring/model versioning; cross-dataset `.mat` loader (`tep_loader.py` extension point); 500-sample slope feature; ablation grid window30/60/90/120 stride2/5/10 latent16/32/64 threshold p99/p99.5/mean+3σ seed42.

**Objectives restated (`PROJECT_REPORT.md:535`):**
1. Reduce stealth miss 80–88%→<40% at FPR<5% without harming easy faults.
2. Exact-ID ≥0.80 (subsystem 1.00), severity ≥0.70.
3. Live demo Streamlit+FastAPI with injected 3/9/15 latency.

**Reproduction (`PROJECT_REPORT.md:568`):**
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
python scripts/prepare_tep.py --config configs/config.yaml --smoke
python scripts/prepare_tep.py --config configs/config_a5000.yaml --full
python scripts/train_anomaly_detector.py --config configs/config_a5000.yaml
python scripts/evaluate_anomaly_detector.py --config configs/config.yaml
python scripts/generate_llm_dataset.py --config configs/config.yaml --samples-per-fault 20
python scripts/train_tep_adapter.py --config configs/config_a5000.yaml
python scripts/test_adapter.py --config configs/config.yaml
python scripts/test_end_to_end.py --config configs/config.yaml --inject-at 800
python scripts/evaluate_all_faults_full.py --config configs/config_a5000.yaml
python scripts/diagnose_fault15_full.py --config configs/config.yaml
python scripts/diagnose_faults_3_9_15.py --config configs/config_a5000.yaml
uvicorn api.server:app --host 0.0.0.0 --port 8000
streamlit run ui/app.py
```
Key config `configs/config.yaml:1` seed42: window60 stride5 float32 block500 test0.2, detector hidden128 layers2 latent64 dp0.05 epochs100 batch64 lr1e-3 percentile99.5, evidence top5 min3% baseline20 trend8 onset1.5 max6, events 3/20/200 db `outputs/events.db`, LLM InternVL2-2B `tep_rca` r16 α32 bf16 batch2×8 ep3 lr2e-4 seq4096 faults train18/val2/test2. **Subtle:** `configs/config_a5000.yaml:62` actually sets `percentile: 99.0` (not 99.5) — a 0.5-point inversion vs `config.yaml:90` `99.5`; the frozen `threshold.json:2` `0.687363` was computed from `config.yaml` p99.5 mean0.619 std0.028; A5000 doc says 99.0 is "less aggressive" but the on-disk A5000 file is *more* aggressive. Treat `threshold.json` as truth, not YAML comment.

---

## 4. PART II — Deepfake GI

> Source: `F:/deepfake gi/LY_C68/deepfake_project/REPORT_exp2_to_exp3.md:1` + source code `model.py:138` (InternVL3-2B LoRA), `dataset.py:71`, `train.py:34`, `eval.py:1`, `inference.py:1`, epoch CSVs, logs, **plus re-audited alternate stacks** `RUN1/model.py` (HydraFakeNet custom), `RUN2/model.py:1` (CLIP ViT-B/32 partial-unfreeze + Transformer 138M), `datasets/model.py:1` (CLIP + FFTBranch + DropPath 131M dual-branch), `datasets/runs/{exp3,exp7,exp_final_single}/train.log` and `runs/intern_exp3_corrective/logs/train.log` (5 resumes). Prior attempt `RUN1/` now fully expanded.

### 4.1 Objectives & Setup

Project `C:\Users\Admin\Desktop\ly project c 68\deepfake_project\deepfake_project` — **deepfake detection + structured reasoning**: classify image real/fake **and** generate auditable chain-of-thought `<fast>…</fast><planning>…</planning><reasoning>…</reasoning><reflection>…</reflection><conclusion>…</conclusion><answer>real|fake</answer>` grounded in visual evidence.

- Base: `InternVL3-2B` (InternViT 448px + Qwen2 LLM + LoRA r64 α128) — `model.py:152`
- Baseline: `runs/intern_exp2` — 11 epochs on `datasets/hydrafake` (InternViT frozen, LoRA on LLM)
- Trusted: `runs/intern_exp2/checkpoints/ep010.pth` (5,166,616,259 bytes ZIP_OK)
- Discarded: `ep011.pth` worse reasoning + `best.pth` 213MB truncated corrupt
- Corrective: `runs/intern_exp3_corrective` via `--weights_from ep010.pth` (`REPORT_exp2_to_exp3.md:8`)
- Date: 2025-08-24+

One-line corrective command (`REPORT_exp2_to_exp3.md:141`):
```bat
python train.py --dataset_root "C:\...\datasets\hydrafake" --json_root "C:\...\datasets\hydrafake\jsons" --output_dir ./runs/intern_exp3_corrective --weights_from ./runs/intern_exp2/checkpoints/ep010.pth --epochs_cls 0 --epochs_joint 2 --lr 1e-5 --lm_loss_weight 0.3 --lm_seq_len 512 --batch_size 4 --grad_accum 8 --num_workers 8 --freeze_cls_head --freeze_projector
```

### 4.2 Dataset — HydraFake

**Structure (`dataset.py:164`, `REPORT_exp2_to_exp3.md:21`):**
- JSONs: `datasets/hydrafake/jsons/train` contains `all.json` 48,320 + 26 per-generator nested 48,320 + `pgrpo_8k.json` 8,033 = **bare** `<answer> X </answer>` (23 chars) → 104,673 (72%) vs `sft_36k.json` 36,750 + `mipo_3k.json` 3,480 = **rich** tagged → 40,230 (28%). Without dedupe 144,903 entries for 48,320 unique images mixed both modes.
- Test mappings: `_TEST_GENERATOR_MAP` in `dataset.py:40` maps 21 generators → 4 categories `cd/cm/cf/id` (e.g., `deepfacelab→cd`, `CodeFormer→cf`, `StyleGAN→id`).
- Image layout: `datasets/hydrafake/train|val|test/{real,fake}/{generator}/img_*.png` with on-disk fix for `train\fake\fake\EFG` duplication.
- Transforms `dataset.py:118`: train = Resize(480)+RandomCrop448+Flip H0.5/V0.1+ColorJitter+Gray0.1+Blur0.3+Rotate10+R_Erasing0.2; val/test = Resize448 CenterCrop.

**Model Architecture (`model.py:36`):**
- `CustomProjector` (`model.py:40`): vision_dim1024 → llm_dim2048 two tokens (CLS + mean-patch) vs old 1-token CLS-only — fix #3.
- `ClassificationHead` (`model.py:89`): vision_dim*2 →256→2, Gelu dropout0.3 — fix #6/#10.
- `DomainRouter` (`model.py:118`): 4-domain MoE stub kept for ckpt compat, not called in forward — fix #8.
- `DeepfakeReasoningModel` (`model.py:138`): LoRA `r64 α128 dropout0.05` on `q/k/v/o gate/up/down` (`model.py:172`), `lm_loss_weight 0.3` (`model.py:146`) fix #9, freeze order LoRA-first then non-LoRA (`model.py:185`) fix #1.

### 4.3 Training History

**RUN1 (`RUN1/runs/exp1/train.log` — early custom HydraFakeNet 60M params):**
- Custom CNN/Transformer (not InternVL) — `RUN1/model.py`, `RUN1/dataset.py`.
- Logs show collapse potential: loss 3.59 (ClsLoss0.70 GenLoss10.33) flat across 4 epochs, val AUC 0.543–0.547, acc ~0.51–0.53 → essentially not learning (see `RUN1/runs/exp1/train.log:144` TRAIN loss 3.5899 acc0.5145). This motivated switch to InternVL3.

**exp2 (`runs/intern_exp2/logs/epochs.csv`):**

| epoch | stage | tr_loss | tr_cls | tr_lm | tr_acc | tr_f1 | va_loss | va_acc | va_f1 | va_auc | best |
|-------|-------|---------|--------|-------|--------|-------|---------|--------|-------|--------|------|
|1|cls_only|0.669|0.669|0|0.624|0.624|0.526|0.765|0.780|0.864|*|
|2|cls_only|0.565|0.565|0|0.724|0.728|0.550|0.741|0.662|0.904|*|
|2|cls_only|0.565|0.565|0|0.723|0.727|0.500|0.785|0.749|0.900|*|
|3|cls_only|0.535|0.535|0|0.752|0.755|0.526|0.767|0.704|0.918|*|
|4|joint|0.757|0.511|0.819|0.776|0.779|0.551|0.762|0.694|0.926|*|
|5|joint|0.706|0.497|0.698|0.788|0.791|0.494|0.798|0.754|0.931|*|
|6|joint|0.675|0.485|0.631|0.799|0.800|0.521|0.788|0.737|0.931| |
|7|joint|0.652|0.477|0.581|0.805|0.807|0.452|0.833|0.811|0.936|*|
|8|joint|0.630|0.467|0.543|0.814|0.813|0.516|0.795|0.748|0.928| |
|9|joint|0.612|0.461|0.503|0.819|0.820|0.429|0.849|0.836|0.941|*|
|10|joint|0.594|0.455|0.463|0.826|0.826|0.417|0.858|0.851|0.942|*|
|11|joint|0.581|0.452|0.430|0.827|0.828|0.484|0.817|0.784|0.946|*|

Peak AUC 0.946 at ep11 but VaAcc dropped 0.858→0.817 and reasoning visibly worse (LaTeX/CJK drift) — hence trusted ep10 not ep11. `best.pth` based on AUC alone picked ep11 and was corrupt 213MB.

Curves saved `runs/intern_exp2/logs/curves.png` (readable via `train.py:118 save_curves`).

**exp3_corrective (`runs/intern_exp3_corrective/logs/epochs.csv:1` + `logs/train.log:1` 187 lines — 5 distinct launches):**

| epoch | stage | tr_loss | tr_cls | tr_lm | tr_acc | va_acc | va_auc | best | log line |
|-------|-------|---------|--------|-------|--------|--------|------|------|----------|
|1|joint|0.712|0.444|0.892|0.834|0.858|0.943|*| `13:41 initial` `lr1e-5 lm0.3 freeze_cls+proj true (73,990,788 3.41%) steps/epoch8841 total17682` |
|1|joint|0.712|0.444|0.892|0.834|0.858|0.943|*| `09:48 resume ep10` identical |
|2|—|—|—|—|—|—|—| | `09:59 resume ep1 best0.943 → total_steps44205 (epochs_joint 5)` — scheduler stretched (patchwork risk) |
|1|joint|0.495|0.427|0.567|—|—|—| | `09:22 from ep001.pth proj-only freeze(12 tensors 3.43%) lr5e-6 lm0.12` — hyperparam sweep |
|1|joint|0.449|0.357|0.759|—|—|—| | `15:07 same proj-only lr5e-6 lm0.12` second run Step400 loss0.402→0.326 |

Note tr_acc 0.834 > exp2 joint 0.776 (same va_acc 0.858, va_auc 0.943 flat by design — classifier frozen). **Subtle:** the log actually contains **5 launches** (two ep10 cold starts, one resume with stretched epochs_joint 2→5 → total_steps 17682→44205, and two lr-halved lm_weight-reduced sweeps from ep001). Only the first 17h run `Ep1 0→8841 0.712` is reflected in `epochs.csv`; the later sweeps are not yet checkpointed as `ep002`. The report previously showed one row — now five are documented. Best remains 0.943.

**Alternate stacks re-audited (not in `REPORT_exp2_to_exp3.md` but present on disk):**

*`RUN2/model.py:1` — CLIP ViT-B/32 + FeatureProjection + ClassificationHead (Transformer 4×512 8 queries) + GRU ReasoningDecoder (138M total, 79M trainable):*

| epoch | TrainLoss | ClsLoss | GenLoss | TrainAcc | ValLoss | ValAcc | AUC | Best |
|-------|-----------|---------|---------|----------|---------|--------|-----|------|
|1|1.956|0.369|5.657|0.852|3.153|0.849|0.935|*|
|5|0.425|0.175|1.009|0.972|4.781|0.878|0.945|*|
|8|0.377|0.165|0.872|0.979|4.628|0.904|0.966|*|
|16|0.313|0.133|0.733|0.993|4.856|0.841|0.911| | early-stop Best 0.9045 (`RUN2/runs/exp1/train.log:72`) |

*`datasets/model.py:1` — CLIP ViT (768) + FFTBranch (FFT magnitude → CNN 32/64/128 → 256) dual-branch fused 512 → ClassificationHead depth2 queries4 DropPath0.1 + ReasoningDecoder inference-only:*

| run | epochs | best ValAcc/AUC | trainAcc final | note |
|-----|--------|-----------------|----------------|------|
| `datasets/runs/exp_final_single` | 45 | **0.8928** /0.955 (`train.log:53` ep17) | 0.990 | Clean convergence, no collapse |
| `datasets/runs/exp3` | 18 then 48 total | 0.8775 /0.958 | 0.997 | Standard scratch model, early-stop 18 |
| `datasets/runs/exp7` (`RUN2/runs/exp1` style cls-only) | 59 | **0.9644** AUC (`train.log:109`) | 0.73 | But `eval.log:1` collapsed to `Accuracy 0.4999 AUC 0.5` (predicts fake only: `pred fake 26633/26633`) — train/eval disconnect — **hidden collapse** |
| `datasets/runs/exp3/eval.log:1` | — | 0.4999 /0.5 | — | Same collapse: 5 repeats all 0.49→0.50 — model learned to predict only `fake` |

This collapse explains why the project pivoted to InternVL3-2B LoRA (`model.py:138`) — scratch CLIP+GRU and FFT dual-branch both hit generalization walls on HydraFake cross-generator splits (`CM` 0.997 vs `CD` 0.796 in `eval_ep010.log:12`).

### 4.4 Eleven Problems Found Since exp2 (`REPORT_exp2_to_exp3.md:21`)

| # | Problem | File:Line | Evidence | Impact |
|---|---------|-----------|----------|--------|
| **P1** | Mixed conflicting targets | `datasets/hydrafake/jsons/train` | 104k bare vs 40k rich, 144k entries for 48k unique, 72% bare | Model learns conflicting modes |
| **P2** | Train/inference mismatch | `dataset.py:71` vs `inference.py:81` | `_extract_reasoning` stripped tags + dropped answer, inference parses tags+answer → raw fallback `answer=unknown` | Eval rambles, tags missing |
| **P3** | No EOS | `dataset.py:219` old | Never appended `tokenizer.eos_token` `<\|im_end\|>` 151645 | Generation ran to `max_new_tokens`, rambled |
| **P4** | Duplicate loading | `dataset.py:165` `glob("**/*.json")` | train 144k/48k, val 8k/4k unique, steps/epoch 18112=144k/8 → 3× time, skewed | Wasted 3× epoch, biased |
| **P5** | Silent black-image fallback | `dataset.py:300` | Missing file → `Image.new(RGB,0)` with label kept; doubled `fake\fake\EFG` | Trained on black images invisibly |
| **P6** | GradScaler BF16 crash | `train.py:47,243` | `GradScaler` FP16-only → RuntimeError ams_foreach…BFloat16 at scaler.unscale Step0 | Crash at start |
| **P7** | Stochastic inference | `model.py:338` | `do_sample True temp0.8 top_p0.9 rep1.3` — only path `inference.py:210 model.predict()` | Forensic outputs unstable |
| **P8** | Wrong log dir + no trim | `inference.py:62` | `default C:\deepfake_project\runs` → logs outside repo; no `trim_after_answer` → 512 ramble | Logs lost, output noisy |
| **P9** | Scheduler patchwork | `runs/intern_exp2/logs/train.log:10` | total_steps 90560 (5ep) for 11ep; 20+ `Resumed from epoch X` rebuilding cosine epochs_joint 2→5→7→10 | LR schedule untrustworthy |
| **P10** | Checkpoint AUC-only | `train.py:525` | `cur_auc>best_auc` → best=ep11 AUC0.946 despite VaAcc .858→.818 worse reasoning | Wrong best |
| **P11** | Windows cp1252 crash | `model.py:228` | `print("… ✓")` → UnicodeEncodeError charmap \u2713 | Crash on Windows |

### 4.5 Changes Made (`REPORT_exp2_to_exp3.md:38`)

#### `dataset.py` — Reasoning Target Overhaul
**Before (`dataset.py:71` old):**
```python
def _extract_reasoning(messages):
    parts = _TAG_RE.findall(content)
    if parts: return " ".join(p.strip() for p in parts)  # tags + answer gone
    return _ANSWER_RE.sub("", content).strip()
```
**After (`dataset.py:71`):**
```python
def _extract_reasoning(messages):
    """Return FULL assistant response verbatim, preserving all tags AND <answer>."""
    for msg in messages:
        if msg.get("role") != "assistant": continue
        return msg.get("content","").strip()
    return ""
```
**`_load` L173–289 two-phase keep-richest:**
1. Phase1 `rich=1 if _TAG_RE.search(raw) else 0`; `candidates: normcase(path)→(rich,resolved,label,raw,ftype)`; later rich wins. Counts Missing(0), Duplicates(96,583).
2. Phase2 `if not rich and split=="train": skip` (11,277 dropped); val/test fallback minimal tagged so `validate()/eval.py` stay 4,000 /52,266 (`dataset.py:227`). EOS appended `target_text=reasoning+tokenizer.eos_token` (`dataset.py:246`); drop >512 `if n_tok>512 skip` never truncates mid-answer (`dataset.py:252`). `max_text_len 192→512` (`dataset.py:153,329`).

**Result verified smoke (`REPORT_exp2_to_exp3.md:64`):** 35,367 usable /48,320 unique / Missing0 / Duplicates96,583 / No-trace11,277 / Too-long1,676 / Fallbacks0; **100% contain `<answer>`+`<fast>`**, mean259 tok 0 truncated.

#### `train.py` — Corrective Mechanics
- Header `train.py:10` 192→512 (95.5% intact).
- One-liner `train.py:34` corrective command added.
- `train.py:47` GradScaler removed → `from torch.amp import autocast` only (`train.py:48`).
- `train.py:343` `lm_seq_len` default 192→512.
- New args `train.py:356` `--weights_from` (critical-key guard `cls_head./custom_projector./lora_`), `--freeze_cls_head`, `--freeze_projector` (`train.py:361`).
- `train_epoch` `train.py:209` scaler removed; `train.py:243` replaced scale/unscale/step/update with `loss.backward()` + `grads=[p for p in parameters if p.grad not None]; clip_grad_norm(grads,1.0); optimizer.step(); scheduler.step()` (`train.py:250`).
- `main` `train.py:417` after `DeepfakeReasoningModel(...).to(device)` → if `weights_from`: load `ckpt["model"]` strict False check critical prefixes → freeze 16 tensors (`cls_head`4 + `custom_projector`12) → log `Trainable 73,990,788 (3.41%)` (`train.py:444`). Optimizer built after freezes LoRA=392 custom=4 dormant (`train.py:457`). Scheduler total_steps `(0+2)*8841=17682` clean (`train.py:478`).

#### `model.py` — Deterministic Inference
- `model.py:228` `✓→[OK]` cp1252 fix.
- `model.py:339` `predict(do_sample=False, temperature1.0, top_p1.0, rep1.05, use_cacheTrue)` was True/0.8/0.9/1.3. Only caller `inference.py:210`.

#### `inference.py` — Auditable Output
- `inference.py:62` `log_dir` hardcoded `C:\deepfake_project` → `default None` + `inference.py:162` resolves `None→script_dir/runs/intern_exp2/logs`.
- New `inference.py:97` `trim_after_answer()` cuts after last `</answer>` + strip `<|im_end|>`.
- `inference.py:228` `raw_text=trim_after_answer(result["reasoning"][0])`.

#### `eval.py` — Unchanged by design (cls-only `validate()`, so lm_seq_len 192 irrelevant); benefits from val fallback fix still 4,000.

### 4.6 Sequence Length 192→512 — Why and Cost (`REPORT_exp2_to_exp3.md:95`)

Measured on fixed keep-richest 35,367 targets (with EOS): `p50 236 / mean272 / p90 457 / p95 507 / p99 576 / max746`.

| Cap | Truncated (lose `<answer>`) | Dropped cleanly | Training time |
|-----|----------------------------|-----------------|---------------|
| 192 old | >256 =14,798 (40%), >384=6,240(17%) | — | 18112×0.76s=3.8h/ep |
| 512 new | **0 truncated** | 1,676 (4.5%) dropped | 8841×2.77s=6.8h/ep (2.67× tokens +7.1× attention →3.7×) 2ep≈13.6h |

Intentional — 196/256 would re-introduce truncated-no-answer teaching.

### 4.7 Will Resuming from ep010 Still Get Better? — Yes, Protected (`REPORT_exp2_to_exp3.md:103`)

| Component | Fate |
|-----------|------|
| Vision (InternViT) | Frozen since day1 untouched |
| Classifier (`cls_head`4 + `custom_projector`12) | **Frozen** `grad check OK: cls_head0 projector0` — cannot degrade |
| Projector | Frozen → LLM prefix stable |
| LoRA 73.8M | Adapts at lr1e-5 vs old 5e-5 to corrected tagged+EOS |
| Base LLM | Frozen bf16 untouched |
| Optimizer/Scheduler | Fresh not restored — old patchwork discarded; total_steps8841 clean |

VaAcc/VaAUC flat by design (~.86/.94) — pick final by LM loss + spot-check not Best.

### 4.8 Verification Completed (`REPORT_exp2_to_exp3.md:129`)

- `py_compile` 4 files SYNTAX OK
- Smoke corrective: 35,367 usable Missing0 Duplicates96,583 No-trace11,277 Too-long1,676; ep010 missing0 VAL4000(2000/2000) grad cls_head0 projector0 lora392
- BF16 crash reproduced+fixed; cp1252 fixed
- `best.pth` 213MB corrupt ZIP vs ep checkpoints ZIP_OK
- All CLI dirs reachable (`datasets/hydrafake`, `/jsons/train|val|test|id|cm|cf|cd`, `train|val|test`, `models/InternVL3-2B`, `runs/intern_exp2/checkpoints/ep010.pth` 5,166,616,259 bytes True)

### Future Scope — Deepfake

| Horizon | Proposal | Rationale |
|---------|----------|-----------|
| Near (1 ep corrective done, 1 more + 2 sweeps) | Finish 2ep corrective **without re-stretching scheduler** (keep total_steps 17682, lr1e-5, lm0.3) and evaluate the two new sweeps (lr5e-6 lm0.12 proj-only) — pick by LM loss not AUC | Current log shows 5 launches with scheduler stretch 2→5 and hyperparam drift; must lock one config to avoid P9 repeat |
| Near | Expand richest-tag dataset to 512 mean272 coverage, drop not truncate | Already done (`dataset.py:153,329`) |
| Medium | Wire `DomainRouter` 4 (`model.py:118`) for MoE ID/CM/CF/CD specialization **or** re-enable FFTBranch path from `datasets/model.py:68` (frequency fingerprints) | Stub needs domain labels from `_TEST_GENERATOR_MAP` `dataset.py:40`; FFT shows promise on `exp_final_single` 0.892 but collapse on cross-domain warns |
| Medium | Full eval on 4 splits with `eval.py` + `inference.py` deterministic + trim — fix `datasets/runs/exp3` collapse (0.499) by re-balancing sampler/ loss | Need `eval_ep010.log:12` style report: ID 0.815/0.900 CM 0.978/0.997 CF 0.779/0.917 CD 0.723/0.796 MEAN 0.824/0.903 |
| Long | Human forensic study: rate reasoning traces for groundedness vs hallucination; also log `runs/intern_exp3_corrective/logs/inference_best_*.log` (3 logs already: `093031`, `093538`, `111437`) | Automated cls metrics miss reasoning quality; those 3 inference logs are the spot-check material |

Artifacts: `runs/intern_exp2/checkpoints/ep010.pth` (5.16GB ZIP_OK) `ep011.pth` `best.pth`213MB corrupt; `logs/train.log` (20 resumes patchwork), `logs/epochs.csv:1` 13 rows, `logs/curves.png`, `logs/eval_ep010.log:12` (ID/CM/CF/CD table above), `logs/inference_*.log`; `runs/intern_exp3_corrective/checkpoints/ep001.pth` + `best.pth` `logs/epochs.csv:1` `logs/steps.csv` `logs/train.log:1` 187 lines 5 launches `logs/inference_best_*.log` ×3; alternate stacks `RUN2/runs/exp1/{train.log,eval.log}` `datasets/runs/{exp3,exp7,exp_final_single}/{train.log,training_curves.png,eval.log}`; `requirements.txt:1` pins `torch 2.6.0+cu126 transformers 4.37.2 peft 0.7.1` (older than anomaly's 4.46).

---

## 5. PART III — Finance RL

> Source: `F:/C68-FinanceRL/RUN_REPORT.md:1` (334 lines, **current truth**) supersedes `REPORT.md:1` (232 lines, stale) — re-audited `RUN_REPORT.md.bak` (identical), `config.py:1`, `config.yaml:1`, `env.py:45`, `model.py:80`, `train.py:36`, `ppo.py`, `data/README.md:1`, `artifacts/finance_prod/eval_2023-01-01_2024-12-31.json:1` (141k/-6.13%), `requirements.txt`. `REPORT.md` quotes retained for legacy but numbers below are from `RUN_REPORT.md`.

### 5.1 Objectives

**Primary (`RUN_REPORT.md:28` supersedes `REPORT.md:24`):**
1. Build portable finance adapter — LoRA (`finance`) that plugs onto shared InternLM backbone beside other adapters without retraining base (`config.py:15` `adapter_name finance`).
2. Learn long-only cost-aware portfolio: fully invested, daily rebalanced, **10 bps transaction cost**, **30% single-asset cap** (`env.py:70` soft-max cap loop, `config.yaml:7` 10.0/0.30).
3. Enforce honest evaluation — train `...→2022-12-31`, test untouched `2023-01-01→2024-12-31` never tuned on (`RUN_REPORT.md:32`).
4. Run efficiently on A5000 (Ampere, sm_86) — `bf16`, `TF32`, fits `24GB`, `75k–100k` production runs `1.3–1.8h`.

**Secondary:** Frozen LLM + tiny encoders can handle numeric time-series (OHLCV + volume + risk); offline data `data/nifty50_prices.csv` bundled (`data/README.md:1` format `Date,Ticker,Close,Volume`); baselines `evaluate_baselines.py`; document pitfalls (survivorship bias, reward vs market).

**Plain language (`RUN_REPORT.md:46` supersedes `REPORT.md:40`):** 49 stocks, 30-day window 5 features (`returns, log-ret, vol, MA gap, norm volume`) ×30×49 + portfolio 49+1 + risk 3 → AI scores 49 → soft-max + cap (`30%` `env.py:70`) → portfolio weights → next-day `gross` minus `cost = turnover×10bps/10,000` → `reward = net -0.001×turnover -0.10×volatility -0.20×drawdown` (`RUN_REPORT.md:58`). No correct answer given; PPO reinforces this reward. **Subtle:** `config.yaml:4` still lists `target_modules: [q_proj, v_proj]` (Llama names) — stale vs `config.py:14` `["wqkv","wo"]` (InternLM fused) which is auto-remapped at runtime (`model.py:75`). Trust `config.py` + `artifacts/.../adapter_config.json` (`wqkv/wo r16`).

### 5.2 Dataset

| Item | Detail |
|------|--------|
| Bundled panel | `data/nifty50_prices.csv` 14MB **287,263 rows** long-format `Date,Ticker,Close,Volume` (`data/README.md:1` format, `Volume` optional) |
| Tickers | 49 Nifty 50 `.NS` (e.g., `RELIANCE.NS`, `TCS.NS`, `INFY.NS` — see `nifty50_summary_statistics.csv` `EICHERMOT 998k%` `BAJFINANCE 236k%`) current list |
| Period | 1999→2026 raw; effective **2017-11-17→2026-01-30** after `pivot().ffill().dropna()` (`env.py:45` + `RUN_REPORT.md:74`) → **2,027 trading days** (`train.py` logs `dates=2027`, honest `1266` after `train-end 2022-12-31` `RUN_REPORT.md:166`) |
| Bias | **Survivorship bias** — current list backfilled; README warns to use NSE archives for rigorous claims (`RUN_REPORT.md:76`) |
| Honest split | Train `...→2022-12-31` ≈1,266 days (`~1,068 steps/episode` with window 30), holdout `2023-01-01→2024-12-31` **459 steps** (2 yrs) (`RUN_REPORT.md:77`) |

Columns: 5 features per day per asset: returns, log-returns, vol, MA gaps, volume (derived in `env.py`).

### 5.3 Model & PPO (`RUN_REPORT.md:85` supersedes `REPORT.md:72` system overview)

```
data/nifty50_prices.csv → PortfolioEnv (window30,10bps,30% cap) → {market,portfolio,risk}
                                                          ↓
                                             FinanceEncoders 3×MLP 2048-d
                                                          ↓ (3 tokens×2048)
                                       InternLM2-1.8B frozen + LoRA finance (wqkv/wo r16)
                                                          ↓ hidden_states[-1][:,-1]
                                       Actor 49→weights          Critic 1→value
                                                          ↓ PPO+GAE ↓
```

- **Backbone:** `internlm/internlm2_5-1_8b-chat` 24 layers hidden2048 bf16 `trust_remote_code`; frozen, only LoRA trainable. Corrected `q_proj/v_proj→wqkv/wo` (`RUN_REPORT.md:96` — auto-remap for legacy) — `config.py:14` `target_modules ["wqkv","wo"]`, `lora_r16 α32 dropout0.05` (`config.py:12`). **Subtle:** `config.yaml:4` still ` [q_proj, v_proj]` stale — trust Python.
- **Encoders:** Market `Linear5→2048→GELU→Linear2048`, Portfolio `LayerNorm50→Linear`, Risk `LayerNorm3→Linear` stacked as 3 tokens `inputs_embeds` (`model.py` + `RUN_REPORT.md:97`); `model.py:75,97` `log_std ~ -0.5` Gaussian.
- **Heads:** Actor `2048→49`, Critic `2048→1`, `log_std` Gaussian.
- **PPO `ppo.py`:** γ0.99 λ0.95 clip0.2 value0.5 entropy0.01 max_grad0.5 4epochs rollout256(prod512) minibatch64(prod128) (`config.py:28` + `RUN_REPORT.md:99`), advantages normalized, `model.py:80` bf16 cast, `train.py:110` bf16→numpy `ppo.py:17` bf16/fp16 branches.
- **Env reward `env.py:63` + `RUN_REPORT.md:58`:** `turnover = |new-old| sum` `cost = turnover×10bps` `net = gross-cost` `reward = net -0.001×turnover -0.10×vol -0.20×drawdown`; info `equity,gross_return,net_return,turnover,trading_cost,drawdown,volatility` (`RUN_REPORT.md:100`).
- **Configs `config.py:6` ModelConfig + `config.py:21` PPOConfig** serializable via `asdict`; `config.yaml:1` 11 lines is truncated duplicate (only `model/reward`).

### 5.4 Hardware & Performance (Verified on RTX A5000) (`RUN_REPORT.md:122` supersedes `REPORT.md:94`)

| Spec | Value |
|------|-------|
| GPU | RTX A5000 WDDM driver 595.95 CUDA13.2 compute sm_86 **24,564MiB (25.8GB)** |
| Torch | **2.11.0+cu128** `cuda:0` True |
| Dtype | **bf16** native + TF32 matmul/cuDNN `high` precision (`train.py:36-40`) |
| VRAM | **3.9GB alloc /6.05 reserved** for 49 assets bf16 — 1.8B ~3.6GB + LoRA/heads |
| Throughput | Eager autocast per step; triton unavailable Win → torch.compile fallback Linux/WSL; smoke512 ~35-70s full25k ~27-45min prod75k ~1.3h (~70 episodes train-split) |
| Logging | Per rollout `mean_reward/mean_net/max_dd/mean_turnover/peak_equity` + VRAM, periodic `[Holdout ...]` learned vs equal-weight when `--train-end` set, `checkpoint_<steps>/` |

File map `RUN_REPORT.md:102` supersedes `REPORT.md:105`: `config.py`, `config.yaml` (duplicate), `env.py`, `model.py`, `ppo.py`, `train.py` (detailed logging + checkpoints + periodic holdout `train.py:45`), `evaluate_policy.py` (new, nested `adapter_config.json` + CUDA), `evaluate_baselines.py`, `prepare_nifty50_data.py`, `data/` (+ `data/README.md:1` + `nifty50_summary_statistics.csv`), `artifacts/` (+ 7 `finance_prod` checkpoints), `tests/` (smoke instructions), `requirements.txt:1` (`yfinance`, `torch≥2.4` but verified `2.11+cu128`).

### 5.5 Progress Report (updated — `RUN_REPORT.md:134` is current truth)

**Timeline recent (`RUN_REPORT.md:134` supersedes `REPORT.md:114` — added #6):**
1. Fixed LoRA target `q_proj/v_proj→wqkv/wo` ValueError + auto-remap.
2. Fixed Transformer5 + InternLM forward `use_cache False output_hidden_states True` extract `hidden_states[-1][:,-1]` (`DynamicCache.from_legacy_cache` removed).
3. Fixed bf16 mismatch encoders `bf16` vs `float32` obs cast `model.py:80`, `train.py:110` bf16→numpy, PPO `float()` GAE.
4. Switched GPU 2.13cpu→2.11cu128 verified cuda sm_86 25.8GB bf16.
5. Added observability `train.py: _metrics()` rollout `mean_net/max_dd/turnover` + `--eval-interval/--checkpoint-interval` periodic holdout.
6. **NEW** Holdout evaluator `evaluate_policy.py` (handles nested `adapter_config.json`, CUDA deterministic mean policy) + `eval_*.json` saving (`RUN_REPORT.md:144`).

**Training evidence (`RUN_REPORT.md:146` supersedes `REPORT.md:122`):**

*Full-panel 25k leaky (no train-end, trained on holdout too) — tail `RUN_REPORT.md:148`:*
```
steps 23552 mean_reward -0.0147 peak 103,087
steps 23808 mean_reward -0.0195 peak 103,087
steps 24064 mean_reward -0.0251 peak 103,087
steps 24320 mean_reward -0.0027 peak 165,188  ← spike one episode
steps 24576 mean_reward -0.0118 peak 189,884
steps 24832 mean_reward -0.0088 peak 105,604  ← give-back
steps 25000 mean_reward -0.0239 peak 105,604
Saved to artifacts/finance_v1
```
Reading: mean_reward penalized negative normal; peak whipsaw 103k→189k→105k high variance — **not yet converged** at 25k (12.5 episodes). Start 100k →105k final high-water +5.6%.

*Honest smoke 512 (train-end 2022-12-31) — with new logging `RUN_REPORT.md:163`:*
```
A5000 config ... dates=1266 | train_end=2022-12-31 ...
steps 256 mean_reward -0.0345 mean_net -0.00115 max_dd 0.3089 mean_turn 1.003 peak 105,299
```
Still exploring (`log_std -0.5`, turnover 1.0 = rewrite daily) but holdout already beats baseline — signal emerges before turnover stabilizes.

*Holdout 2023-01-01→2024-12-31 (459 days) — **4-way** table (`RUN_REPORT.md:184` supersedes `REPORT.md:142` 3-way):*

| Run | How trained | Saw 2023-24? | **Final equity** | **Profit on ₹1L** | **vs equal-weight** |
|-----|-------------|--------------|------------------|-------------------|---------------------|
| **A. Equal-weight (1/49)** | No training (rule) | N/A | **₹1,50,471** | **+₹50,471 (+50.47%)** | — bar |
| **B. AI honest 512** (`smoke_honest/checkpoint_512`, `train-end 2022-12-31`) | 512 steps honest | **No** | **₹1,51,187** | **+₹51,187** | **+₹715 (+0.47% alpha)** |
| **C. AI leaky 25k** (`finance_v1`, no train-end) | 25k steps leaky | **Yes** | **₹1,57,138** | **+₹57,138** | **+₹6,666 (+4.43%) — cheated** |
| **D. AI honest longer** (`artifacts/finance_prod/eval_2023-01-01_2024-12-31.json:4` `141,251`) | Honest longer (70k+ steps, 7 checkpoints 10240→70144) | **No** | **₹1,41,251** | **+₹41,251** | **-₹9,220 (-6.13% alpha)** |

Detailed 459-step metrics `RUN_REPORT.md:204` (all same costs):

| Metric (daily avg) | A. Equal-weight | B. Honest 512 | C. Leaky 25k | D. This honest longer |
|--------------------|----------------|-------------|-------------|-----------------------|
| **Final equity** | **150,471** | **151,187** | **157,138** | **141,251** |
| **Peak equity** | 170,923 | 169,877 | 168,430 | **172,854** |
| **Total return (2 yr)** | 50.47% | 51.19% | **57.14%** | 41.25% |
| **Mean net / day** | **0.0919%** | 0.0934% | **0.1025%** | 0.0852% |
| **Mean gross / day** | 0.0921% | **0.1033%** | 0.1028% | 0.0860% |
| **Turnover / day** | **0.0022** | **0.0989** | **0.0022** | 0.0082 |
| **Max drawdown** | **12.47%** | 12.99% | **10.15%** | **22.44%** |
| **Daily vol** | **0.0076** | 0.0081 | 0.0090 | **0.0140** |
| **Sharpe proxy** | **1.93** | 1.82 | 1.81 | 0.96 |

Interpretation (`RUN_REPORT.md:220` supersedes `REPORT.md:148`): B edged baseline despite turnover 0.099; C flatters via leakage; D fell behind because it learned **too-volatile pre-2023 pattern** (vol 0.014 1.85× baseline, DD 22.44% 172k→133k, gross 0.086% < baseline 0.092% — not just cost but stock selection worsened). Suggests penalty tuning (`turnover_penalty 0.001→0.003` or `max_weight 0.30→0.20`, `lr 3e-4→1e-4`).

**Artifacts (`RUN_REPORT.md:172` supersedes `REPORT.md:155`):**
- `artifacts/finance_v1/{finance_lora/finance/adapter_config.json + adapter_model.safetensors, ppo_heads.pt, model/ppo/environment_config.json, eval_2023-01-01_2024-12-31.json:1}` — confirmed 157k.
- `artifacts/smoke_honest/checkpoint_{256,512}/{finance_lora/finance/, ...}` + `eval_...json` — proven nested loading.
- **NEW** `artifacts/finance_prod/` now 7 checkpoints (`10240,20480,30208,40448,50176,60416,70144` + final `finance_lora/finance/adapter_config.json`) + `eval_2023-01-01_2024-12-31.json:4` learned 141,251 peak172,854 vol0.014 Sharpe0.96 — the `-6.13%` run. Plus `artifacts/finance_test/` and `RUN_REPORT.md.bak` (identical to `RUN_REPORT.md`).
- `nifty50_summary_statistics.csv` per-ticker wow factor (`EICHERMOT 998k%`).

**Key insights (`RUN_REPORT.md:239` supersedes `REPORT.md:162` + two new):**
- Negative reward ≠ losing money (`reward = net -0.10×vol -0.001×turnover -0.20×DD`).
- Peak equity is watermark not final wealth.
- Honest splits matter — leaky flatters (+4.43 vs +0.47 → -6.13 shows same arch can look good/bad).
- **NEW** Early honest already beats baseline; longer honest can still fail if it over-learns volatile pre-2023 pattern (vol 0.014).
- A5000 under-used 3.9GB → `rollout 512 / minibatch 128` fits easily.
- **NEW** Metric gap: `gross 0.086%` < baseline suggests penalty tuning needed, not just turnover.

### 5.6 Limitations Today & Future Scope (`RUN_REPORT.md:249` is current truth)

**Limitations (`RUN_REPORT.md:249` supersedes `REPORT.md:172` — adds one):**
- Needs **75–100k honest steps** to stabilize Turnover/DD; `25k` dev-scale, `512` smoke, **this longer honest run still under-regularized** (DD 22% vs 12% baseline).
- Panel has **survivorship bias** (current Nifty 50 backfilled); pre-2017 rows synthetic (`RUN_REPORT.md:252`).
- Long-only fully invested daily — no shorts/leverage/intraday/order-book.
- `torch.compile` needs Triton Linux/WSL not Windows A5000.
- No live/broker integration by design (and not planned).

**Near-term 1–2m — What We Will Run Next (`RUN_REPORT.md:262` supersedes `REPORT.md:184`):**
- **Controlled honest production:** `75k–100k` on `train-end 2022-12-31` (`70–94` episodes, `1.3–1.8h`, `rollout 512 / minibatch 128 / bf16`, eval `5k`, ckpt `10k`) — *this longer run was that attempt, showing penalty needs tuning before scaling further*.
- **One knob at a time ablations:** `turnover_penalty 0.001→0.003` or `max_weight 0.30→0.20` to calm churn; `lr 3e-4→1e-4`, `clip 0.2→0.15`, `entropy 0.01→0.005` to calm PPO (`RUN_REPORT.md:265`).
- **Early-stop rule:** promote to `100k` only if `mean_net` slope at `60k` stays positive; stop if flat `15k` (`RUN_REPORT.md:266`).
- **Logging:** TensorBoard/CSV `reward / net / gross / turnover / vol / drawdown / Sharpe` (info dict already has all `RUN_REPORT.md:267`).

**Medium 2–6m (`RUN_REPORT.md:269`):** Historical NSE constituent archives → remove survivorship bias; add splits/dividends adjusted; richer state sector/position-level/earnings-calendar/macro via InternLM text; multi-adapter `finance`+`atc` routing; risk suite Sortino/Calmar/turnover-aware Sharpe/short/delta-neutral; Linux/WSL Triton 15–25% speedup.

**Longer-Term Vision (`RUN_REPORT.md:277`):** Portable finance foundation small LoRAs shareable/mergeable without retraining 1.8B core; walk-forward regime tests + proper out-of-sample `2025-26` test once `2023-24` locked as validation; notebooks 3-command `prepare→train→evaluate` (`RUN_REPORT.md:281`).

**Reproduce A5000 honest — copy-paste (`RUN_REPORT.md:285` supersedes `REPORT.md:208`):**
```powershell
# 1) CUDA torch (driver 595.95 / CUDA 13.2 → cu128) — already verified 2.11+cu128
pip install torch --index-url https://download.pytorch.org/whl/cu128

# 2) Smoke honest (~30–70s, proves pipeline, honest)
python train.py --data data\nifty50_prices.csv --train-end 2022-12-31 --timesteps 512 --output artifacts\smoke_a5000 --device cuda --dtype bf16 --eval-interval 256
python evaluate_policy.py --data data\nifty50_prices.csv --artifacts artifacts\smoke_a5000 --start 2023-01-01 --end 2024-12-31 --device cuda
python evaluate_baselines.py --data data\nifty50_prices.csv --start 2023-01-01 --end 2024-12-31

# 3) Production honest (75k ≈70 episodes ~1.3h; 100k ≈94 episodes ~1.8h) — honest
python train.py --data data\nifty50_prices.csv --train-end 2022-12-31 --timesteps 75000 --output artifacts\finance_prod --device cuda --dtype bf16 --rollout-steps 512 --minibatch-size 128 --eval-interval 5000 --checkpoint-interval 10000
python evaluate_policy.py --data data\nifty50_prices.csv --artifacts artifacts\finance_prod --start 2023-01-01 --end 2024-12-31 --device cuda
python evaluate_policy.py --data data\nifty50_prices.csv --artifacts artifacts\finance_prod --start 2025-01-01 --end 2026-01-30 --device cuda  # second unseen

# 4) Leaky vs honest comparison (same holdout)
python evaluate_policy.py --data data\nifty50_prices.csv --artifacts artifacts\finance_v1 --start 2023-01-01 --end 2024-12-31 --device cuda
python evaluate_policy.py --data data\nifty50_prices.csv --artifacts artifacts\smoke_honest\checkpoint_512 --start 2023-01-01 --end 2024-12-31 --device cuda
# this honest longer run
python evaluate_policy.py --data data\nifty50_prices.csv --artifacts artifacts\finance_prod --start 2023-01-01 --end 2024-12-31 --device cuda
```
Outputs `artifacts/.../{finance_lora, ppo_heads.pt, model/ppo/environment_config.json, eval_*.json, checkpoint_*}`. Glossary `RUN_REPORT.md:318` defines LoRA/turnover/drawdown/Sharpe/honest-leaky for PPT.

---

## 6. PART IV — Cross-Cutting Synthesis

### 6.1 What Was Being Built Across All Three?

All three are **frozen foundation model + tiny trainable adapter** applied to a different modality, evaluated with **honesty guarantees**:

| Domain | Foundation frozen | Adapter | Honesty mechanism | Adapter size |
|--------|-------------------|---------|-------------------|--------------|
| Anomaly | InternVL2-2B (InternLM2-Chat-1.8B + ViT) | `tep_rca` LoRA r16 | Detector: normal-only; LLM: fault-disjoint 1008/40/40 | ~62MB `adapter_model.safetensors` |
| Deepfake | InternVL3-2B (Qwen2 + InternViT) | LoRA r64 73.9M 3.41% | Keep-richest dedupe; val splits tagged-fallback 4000 | 73M |
| Finance | InternLM2.5-1.8B-chat | `finance` LoRA r16 on wqkv/wo | Chronological cutoff 2022-12-31 holdout 459 steps | small + heads |

The **meta-lesson** repeated three times: *do not let the model cheat by seeing the test regime through data leakage, adapter leakage, or metric selection.*

### 6.2 Past — What Was Attempted, What Went Wrong, What Went Right

**Anomaly past:** Scaffolding → preprocessing leak-free → detector training succeeded but threshold 99.0 gave FPR13.5% → pushed to 99.5; LLM dataset small (8 per fault) — succeeded but severity weak. **Went wrong:** didn't anticipate stealth class; threshold trade-off hidden until 10k-run sweep. **Went right:** rigorous manifests and per-run CSVs made diagnosis possible without retraining.

**Deepfake past:** RUN1 custom HydraFakeNet 60M collapsed (AUC0.54). Switch to InternVL3-2B vaulted to 0.86→0.94 AUC in exp2. **Went wrong:** 11 bugs accumulated over ~20 resumes (mixed bare/rich targets 72/28, no EOS, 3× duplication, BF16 crash, stochastic decoding, scheduler patchwork, AUC-only best, cp1252). **Went right:** epoch 10 peak was strong; corrective recipe protects all gains by freezing vision+classifier.

**Finance past:** Multiple bugs (wqkv target, hidden_states extraction, bf16 cast) fixed sequentially; 25k leaky run flattered +4.4%; smoke honest proved pipeline but showed high turnover early. **Went right:** bundled data allows offline clone; equal-weight baseline sanity-checks every run.

**Common failure archetypes:**
1. **Leakage / confounding** — Anomaly hypothetical scaler leakage (prevented), Deepfake duplication/merged bare traces teaching two voices, Finance chronological leakage.
2. **Representation mismatch** — Anomaly 60-window smooths drift; Deepfake 192 truncates answer; Finance 30-day window cannot see regime.
3. **Metric gaming** — Anomaly window-recall hides event-miss 80%; Deepfake AUC-only picks ep11 despite reasoning collapse; Finance peak_equity misread as profit.
4. **Engineering brittleness** — BF16 GradScaler crash (Deepfake P6, Finance fixed via `torch.amp autocast`), cp1252, path hardcoding.

### 6.3 Present — Where Each Domain Stands Today (Sep 2026 snapshot)

| Domain | Working pipeline? | Production checkpoint | Honest holdout metric | Known limitation |
|--------|-------------------|-----------------------|-----------------------|------------------|
| **Anomaly** | Yes — `main.py: TEPApp` streams 52 sensors → event → evidence → LLM JSON → SQLite -> API/UI | `outputs/anomaly_detector/model.pt` 2MB thr0.687 + `outputs/tep_rca_adapter/tep_rca/` 62MB | Window F1 0.861 AUROC0.886 delay6.25; LLM subsystem1.00 | Stealth 3/9/15 miss 80–88% |
| **Deepfake** | Yes — `train.py:34` corrective + `inference.py` deterministic + `eval.py` 4 splits; **plus** alternate CLIP+FFT stacks `RUN2/` `datasets/` | `runs/intern_exp2/checkpoints/ep010.pth` 5.16GB trusted (eval `ID 0.815 CM 0.978 CF 0.779 CD 0.723 MEAN 0.824`) + `runs/intern_exp3_corrective/checkpoints/ep001.pth` (1/2 ep + 2 sweeps lr5e-6) | Val AUC 0.943 (frozen) but alternate stacks collapsed 0.499 | Reasoning format corrected but scheduler stretched 2→5; need lock |
| **Finance** | Yes — `train.py` PPO bf16 + `evaluate_policy.py` + `evaluate_baselines.py` | `artifacts/finance_v1` (leaky +4.43% 157k) + `smoke_honest/checkpoint_512` (honest +0.47% 151k) + `artifacts/finance_prod` (7 ckpts 10240→70144 honest **-6.13% 141k DD22.44% vol0.014 Sharpe0.96**) | Honest 512 +0.47% vs honest longer -6.13% — penalty tuning needed | Survivorship bias + under-regularized longer run |

### 6.4 Future — What Is Planned, What Is Proposed, What Is Needed

Unified near-term roadmap (1–2 months) across all:

| Domain | Next milestone | Success criterion | Effort |
|--------|---------------|-------------------|--------|
| Anomaly | Multi-scale windows + prediction head + ROC sweep on frozen detector **(fix `config_a5000.yaml:62` 99.0 vs `config.yaml:90` 99.5 confusion, use `threshold.json` 0.687)** | Stealth miss <40% at FPR<5% without harming 17 easy | No retraining; new scoring heads |
| Deepfake | Finish exp3_corrective epoch2 **without scheduler stretch** (lock 17682, lr1e-5, lm0.3), eval 4 splits `eval_ep010.log:12`, human rating + spot-check `inference_best_*.log` ×3 | Reasoning traces coherent, no 0.499 collapse, AUC stable | 1 epoch + eval |
| Finance | **Single-knob ablation before scaling** (`turnover_penalty 0.001→0.003` or `max_weight 0.30→0.20`) then 75k honest production run 1.3h, TensorBoard | Honest longer stops -6.13% DD22% → turnover/vol calm, alpha ≥0 | Single ablation + long run |

Medium-term (2–6 months): constrained AE / SVDD for anomaly; DomainRouter MoE for deepfake; NSE archival constituents for finance; shared Infra: A5000 bf16, LoRA portability, SQLite/event store patterns reusable.

Long-term vision: **Portable LoRA library on shared InternLM backbone** — `tep_rca` (chemical plant forensic), deepfake forensic, `finance` (portfolio) all loadable as named adapters without retraining core; walk-forward / regime / human evaluation harnesses; documentation notebooks 3-command reproduce.

### 6.5 Inter-Domain Dependencies & Transferable Lessons

- **LoRA correctness order** (`model.py:185`) LoRA-first then freeze — anomaly tep_rca and deepfake both use this; finance wqkv fix analogous.
- **EOS / termination learning** — deepfake P3 teaches that any generative adapter must learn when to stop; anomaly LLM uses `max_new_tokens 768` but evidence-JSON-constrained; finance Actor uses `log_std` Gaussian but clipping clip0.2 analogous stopping.
- **Honest splits by semantic ID not random** — anomaly by fault ID, deepfake by richest-tag keep-one-per-image, finance by date. All reject random splits that leak.
- **Evaluation at the right granularity** — anomaly window vs event, deepfake token-level LM loss vs AUC (P10 lesson: don't pick Best by AUC alone), finance stepwise mean_reward vs holdout annual Sharpe.
- **Negative results are artifacts** — anomaly prediction/relationship negative studies and deepfake RUN1 collapse logs are saved and cited; they narrow search space.

---

## 7. Artifacts Inventory

### 7.1 Tables Embedded Above (all from source reports / code)

- Anomaly window-level metrics (F1 0.861, FPR1.42% etc.) — `PROJECT_REPORT.md:225` + `data/processed/anomaly_detector_eval.json:1` (threshold 0.687364023… FPR 0.01423)
- Anomaly per-fault event table (20 faults) — `PROJECT_REPORT.md:248`
- Anomaly sensor forensics Fault15 (XMEAS_11 ratio1.21) — `PROJECT_REPORT.md:293`
- Anomaly LLM adapter metrics (1.00/0.50) — `PROJECT_REPORT.md:305` + `outputs/tep_rca_adapter/evaluation.json:5` (n40 root0.5 severity0.325)
- Anomaly 5-stage Fault15 diagnosis — `PROJECT_REPORT.md:337`
- Anomaly all-faults grouping Easy/Hard — `PROJECT_REPORT.md:370`
- Anomaly comparative 3/9/15 vs Normal — `PROJECT_REPORT.md:397`
- Anomaly prediction vs reconstruction, relationship summaries — `PROJECT_REPORT.md:430,444`
- Anomaly checklist & timeline — `PROJECT_REPORT.md:181,197` + subtle `config_a5000.yaml:62` 99.0 vs `config.yaml:90` 99.5 + `training_summary.json:9` `wqkv/wo/w1/w2/w3`
- Deepfake 11-problems table P1–P11 — `REPORT_exp2_to_exp3.md:21`
- Deepfake seq length 192→512 study (p50 236 etc. + time) — `REPORT_exp2_to_exp3.md:95`
- Deepfake epochs.csv exp2 (11 rows) + exp3 (1 row + 5 launches) — `runs/intern_exp2/logs/epochs.csv:1`, `runs/intern_exp3_corrective/logs/epochs.csv:1` + `logs/train.log:1` 187 lines 5 resumes
- Deepfake corrective weight fate table — `REPORT_exp2_to_exp3.md:103` updated with freeze 73M/74M sweep
- Deepfake alternate stacks — `RUN2/runs/exp1/train.log:55` best 0.9045, `datasets/model.py:1` FFTBranch, `datasets/runs/exp_final_single/train.log:17` 0.8928, `datasets/runs/exp3/eval.log:1` collapse 0.4999, `runs/intern_exp2/logs/eval_ep010.log:12` 4-split table ID 0.815 CM 0.978 CF 0.779 CD 0.723
- Finance holdout 4-way table (150k/151k/157k/141k) — `RUN_REPORT.md:184` supersedes `REPORT.md:142` 3-way
- Finance detailed 459-step metrics (turnover 0.0082 DD22.44% vol0.014 Sharpe0.96) — `RUN_REPORT.md:204` + `artifacts/finance_prod/eval_2023-01-01_2024-12-31.json:4` `141251` + `artifacts/finance_v1/eval_...json:4` `157137`
- Finance hardware spec (A5000 25.8GB etc.) — `RUN_REPORT.md:122` supersedes `REPORT.md:94`
- Finance training trace last 1.5k steps + smoke 256 `mean_turn 1.003` — `RUN_REPORT.md:148,166`
- Cross-domain comparison + synthesis tables — new + §7.5 subtle deltas

### 7.2 Graphs / Images — Where to Find Them

| Description | Path | Status | How to view |
|-------------|------|--------|-------------|
| Detector performance spectrum placeholder (17 easy vs 3 stealth) | `PROJECT_REPORT.md:245` placeholder `via.placeholder.com/800x200` | Placeholder in report | Regenerate via `evaluate_all_faults_full.py` |
| Event rate vs Mean max scatter (ASCII in report `PROJECT_REPORT.md:278`) | ASCII art in markdown | Embedded | Copy ASCII above |
| Score distribution per fault (CSV-driven) | `outputs/evaluation/fault15_score_distribution.csv` (anomaly) if exists, else generated by `diagnose_fault15_full.py` | CSV artifact | Plot with `python -c "import pandas; df=pd.read_csv('...'); df.hist()"` |
| Training curves PNG (Loss/Acc/AUC) | `F:/deepfake gi/LY_C68/deepfake_project/runs/intern_exp2/logs/curves.png` + `runs/intern_exp3_corrective/logs/curves.png` (via `train.py:118 save_curves`) + `F:/deepfake gi/LY_C68/deepfake_project/datasets/runs/exp_final_single/training_curves.png` + `datasets/runs/exp3/training_curves.png` + `datasets/runs/exp7/training_curves.png` | Exists | Open PNG |
| Epoch CSV → curves | `runs/intern_exp2/logs/epochs.csv`, `runs/intern_exp3_corrective/logs/epochs.csv` (13 +1 rows + 5 launches) | Exists | Regenerate PNG via `save_curves()` |
| Anomaly evaluation PNGs (if any) | `outputs/evaluation/*.png` (none checked in; generated on demand by diagnostics) | Generated by `diagnose_*` scripts | Run diagnostics |
| Finance equity curves (logged, not PNG) | `artifacts/*/logs/` + TensorBoard proposed (`RUN_REPORT.md:267`) + ASCII ₹ chart `RUN_REPORT.md:193` (₹1L→₹1.41–1.57L) | CSV/JSON; TB planned | Plot `eval_*.json` final_equity peak_equity |

> To regenerate visuals without retraining: run `python scripts/evaluate_all_faults_full.py --config configs/config_a5000.yaml` (anomaly, ~15min CPU, writes `all_faults_detector_summary.csv` + heatmaps if plotting enabled) and `save_curves()` in `train.py:118` reads `epochs.csv` → `curves.png`.

### 7.3 Checkpoints & Weights (frozen, do not retrain to verify)

| Domain | Path | Size / Count | Integrity |
|--------|------|--------------|-----------|
| Anomaly detector | `anomaly gi/C68-Anomaly/outputs/anomaly_detector/model.pt` | 2 MB | Produced by `train_anomaly_detector.py` |
| Anomaly threshold | `outputs/anomaly_detector/threshold.json` | thr0.687 mean0.619 std0.028 | `p99.5` |
| Anomaly adapter | `outputs/tep_rca_adapter/tep_rca/{adapter_config.json,adapter_model.safetensors}` + `training_summary.json` | ~62MB | Peft LoRA r16 |
| Deepfake trusted | `deepfake gi/LY_C68/deepfake_project/runs/intern_exp2/checkpoints/ep010.pth` | 5,166,616,259 bytes | ZIP_OK |
| Deepfake discarded | `ep011.pth` (reasoning drift), `best.pth` 213MB truncated corrupt | — | Marked corrupt |
| Deepfake corrective | `runs/intern_exp3_corrective/checkpoints/ep001.pth` + `logs/epochs.csv` | ~5GB + 1 row | Trainable 73M 3.41% |
| Finance honest smoke | `C68-FinanceRL/artifacts/smoke_honest/checkpoint_512/{finance_lora/finance/adapter_config.json, ppo_heads.pt, eval_*.json}` `151,187/+0.47%` + `checkpoint_256` | small | `adapter_config.json` wqkv/wo r16 (verified `smoke_honest/checkpoint_512/eval_*.json:4`) |
| Finance leaky | `artifacts/finance_v1/{finance_lora/finance/, eval_*.json}` `157,137/+4.43%` `model_config.json` `ppo_config.json` | — | Leaky but eval JSON valid (`artifacts/finance_v1/eval_...json:4`) |
| Finance honest longer | `artifacts/finance_prod/{finance_lora/finance/adapter_config.json, ppo_heads.pt, eval_2023-01-01_2024-12-31.json:4` `141,251/-6.13%` `peak172,854 vol0.014 DD22.44%` + 7 checkpoints `10240,20480,30208,40448,50176,60416,70144` each with `finance_lora/finance/adapter_config.json` | 7+final | New `-6.13%` run supersedes 25k leaky illusion |
| Finance test | `artifacts/finance_test/{finance_lora/finance/adapter_config.json}` | — | Smoke harness |
| Data panels | `C68-FinanceRL/data/nifty50_prices.csv` 14MB 287k rows `nifty50_summary_statistics.csv` `data/README.md:1` (format spec); `anomaly gi/.../data/raw/*TEP*` 250k+5M rows + `data/llm/split_metadata.json:1` 1008/40/40 | — | Offline bundled |

### 7.4 Diagnostic & Eval Outputs (per-report inventory)

**Anomaly (`PROJECT_REPORT.md:602`):**
- `data/processed/anomaly_detector_eval.json` window metrics
- `outputs/anomaly_detector/threshold.json`
- `outputs/evaluation/all_faults_detector_summary.csv` per-fault 20 rows
- `outputs/evaluation/all_faults_detector_per_run.csv` per-run 10,075 rows
- `outputs/evaluation/fault15_diagnostic_report.md` 6-stage
- `outputs/evaluation/fault15_sensor_reconstruction.csv` 52 rows
- `outputs/evaluation/faults_3_9_15_final_report.md`
- `outputs/evaluation/prediction_vs_reconstruction_summary.csv` negative
- `outputs/evaluation/relationship_detector_summary.csv` 73 rows
- `outputs/tep_rca_adapter/evaluation.json` 40 samples
- `outputs/tep_rca_adapter/training_summary.json`
- `outputs/llm_dataset_v2/dataset_stats.json` 400 synthetic

**Deepfake (re-audited complete):**
- `runs/intern_exp2/{checkpoints/ep010.pth 5.16GB, ep011.pth, best.pth 213MB corrupt}` + `logs/train.log` (~20 resumes, patchwork `total_steps 90560` for 11ep) + `logs/epochs.csv:1` 13 rows + `logs/steps.csv` + `logs/curves.png` + `logs/eval_ep010.log:12` (ID 0.815/0.900 CM 0.978/0.997 CF 0.779/0.917 CD 0.723/0.796 MEAN 0.824/0.903) + `logs/eval_ep00{1,2,4,6,7,8}.log` + `logs/inference_ep*.log` ×8 + `runs/inference_logs/inference_*.log` ×2
- `runs/intern_exp3_corrective/{checkpoints/ep001.pth + best.pth}` + `logs/train.log:1` 187 lines **5 launches** (lr1e-5→5e-6 lm0.3→0.12 freeze 16→12 tensors 73M→74M) + `logs/epochs.csv:1` 1 row (only first launch) + `logs/steps.csv` + `logs/inference_best_20260831_*.log` ×3 (`093031`, `093538`, `111437`)
- `RUN1/{model.py, dataset.py, train.py, inference.py, utils.py, transformer_blocks.py}` + `RUN1/runs/exp1/train.log:144` early collapse `loss3.5899 acc0.514` val AUC0.54
- `RUN2/{model.py:1 138M CLIP ViT-B/32 + Transformer 4×512 + GRU, dataset.py, train.py, evaluate.py, inference.py}` + `RUN2/runs/exp1/train.log:55` best 0.9045 (16ep early-stop) + `RUN2/runs/exp1/eval/eval.log`
- `datasets/{model.py:1 585 lines CLIP 256 + FFTBranch 256 → 512 DropPath, dataset.py, evaluate.py, inference.py, train.py, utils.py, test_images.py}` + `datasets/runs/exp3/{train.log:1 48ep best0.8775, training_curves.png, eval/eval.log collapse 0.4999}` + `datasets/runs/exp7/{train.log 59ep best AUC0.9644, training_curves.png, eval/eval.log 0.4999}` + `datasets/runs/exp_final_single/{train.log 45ep best0.8928, training_curves.png 42k, eval/eval.log}` + `datasets/runs/exp{5,6,8}/train.log`
- `requirements.txt:1` `torch 2.6.0+cu126 torchvision 0.21 transformers 4.37.2 peft 0.7.1` (pin differs from anomaly 4.46)
- `ppt content.txt:1` empty (placeholder for slides)

**Finance (re-audited current truth = `RUN_REPORT.md:1`):**
- `RUN_REPORT.md:1` 334 lines + `RUN_REPORT.md.bak` (identical) + `REPORT.md:1` 232 lines (stale, superseded) + `data/README.md:1` (format spec) + `config.yaml:1` 11 lines stale `[q_proj,v_proj]` vs `config.py:14` `wqkv/wo`
- `artifacts/smoke_honest/checkpoint_{256,512}/eval_2023-01-01_2024-12-31.json` (151,187/+0.47%), `artifacts/finance_v1/eval_...json` (157,137/+4.43%), `artifacts/finance_prod/eval_2023-01-01_2024-12-31.json:1` (**141,251/-6.13% peak172,854 vol0.014 DD22.44% Sharpe0.96 459 steps**)
- `artifacts/finance_prod/checkpoint_{10240,20480,30208,40448,50176,60416,70144}/` each `finance_lora/finance/adapter_config.json` + `model_config.json` `ppo_config.json` `environment_config.json` + final `artifacts/finance_prod/{finance_lora/finance/adapter_config.json, model_config.json, ppo_config.json, environment_config.json}`
- `artifacts/finance_test/` + `artifacts/.gitkeep`
- `requirements.txt:1` `torch≥2.4 yfinance 0.2.54` but verified `torch 2.11+cu128`; `tests/README.md` smoke instructions

### 7.5 Subtle On-Disk Deltas Since Last Revision (Complete Go-Through)

> This section is the delta audit requested — every file re-globbed (`anomaly gi` 60 files, `deepfake gi` 100+ files, `C68-FinanceRL` 60+ files). No code re-run, only file content diff.

| # | Domain | File | Subtle Change | Impact on Report | Action Taken |
|---|--------|------|---------------|------------------|--------------|
| 1 | Anomaly | `configs/config_a5000.yaml:62` `percentile: 99.0` vs `configs/config.yaml:90` `99.5` | Inversion: A5000 file is *more* aggressive (lower threshold) despite comment "less aggressive"; `threshold.json:2` `0.687363` matches 99.5 | Report previously said A5000 99.5; now clarifies truth is `threshold.json` | Edited §3.4.2, §3.5 future, §7.1 |
| 2 | Anomaly | `outputs/tep_rca_adapter/training_summary.json:9` `["wqkv","wo","w1","w2","w3"]` vs `configs/config.yaml:156` `["q_proj",…]` | Doc vs reality: adapter actually uses InternLM fused names + FFN `w1/w2/w3` (5 targets, not 7) | Report previously quoted YAML 7 targets | Edited §3.4.5 to quote JSON as authoritative |
| 3 | Anomaly | `data/processed/anomaly_detector_eval.json:2` `threshold 0.687363147… precision 0.9999569 FPR 0.01423` and `data/llm/split_metadata.json:34` `n_train 1008 n_val 40 n_test 40` | Exact floats vs rounded 0.687 in narrative | Precision matters for threshold sweep proposal | Added exact values §3.6, §7.1 |
| 4 | Deepfake | `runs/intern_exp3_corrective/logs/train.log:1` 187 lines — 5 launches (not 1) | Previously showed 1 epoch; log shows 5: two cold `lr1e-5 lm0.3 freeze16`, one resume stretched `epochs_joint 2→5 total_steps17682→44205`, two sweeps `lr5e-6 lm0.12 freeze12` from `ep001.pth` | Report was incomplete on corrective progress | Expanded §4.3 table to 5-row log |
| 5 | Deepfake | `RUN2/model.py:1` (138M CLIP ViT-B/32 partial-unfreeze + GRU) + `RUN2/runs/exp1/train.log:55` (16ep best 0.9045) | Alternate stack never mentioned; shows shallow scratch can hit 0.904 but not cross-domain | Complete go-through requires covering `RUN2/` | Added full subsection §4.3 alternate stacks |
| 6 | Deepfake | `datasets/model.py:1` (585 lines CLIP 256 + FFTBranch 256 DropPath) + `datasets/runs/exp3,exp7,exp_final_single/train.log` | New dual-branch frequency-aware model (256+256 fused) — not in `REPORT_exp2_to_exp3.md`; `exp_final_single` clean 0.8928 vs `exp3` collapse 0.4999 | Report omitted FFT work | Added §4.3 + §4.8 future FFTBranch option |
| 7 | Deepfake | `runs/intern_exp2/logs/eval_ep010.log:12` 4-split table `ID 0.815/0.900 CM 0.978/0.997 CF 0.779/0.917 CD 0.723/0.796 MEAN0.824` + `datasets/runs/exp3/eval.log:1` 0.4999 collapse | Actual eval numbers vs narrative "0.824/0.903" — now sourced line-accurate; collapse shows generalization failure | Report previously not quoting ID/CM/CF/CD | Added to §4.3 and §7.2 |
| 8 | Deepfake | `datasets/runs/exp7/train.log:109` best AUC 0.9644 but `datasets/runs/exp7/eval/eval.log:1` accuracy 0.4999 (predicts fake only) | Train best vs test collapse discrepancy — hidden failure mode | Report previously missed | Added to §4.3 alternate stacks |
| 9 | Deepfake | `requirements.txt:1` pins `torch 2.6.0+cu126 transformers 4.37.2 peft 0.7.1` vs anomaly `4.46.3` | Version pin subtle; anomaly uses newer transformers | Report previously unified | Added to §7.4 |
| 10 | Finance | `RUN_REPORT.md:1` 334 lines supersedes `REPORT.md:1` 232 lines + `RUN_REPORT.md.bak` + `data/README.md:1` + `artifacts/finance_prod/eval_2023-01-01_2024-12-31.json:4` `141251/-6.13% DD22.44%` + 7 checkpoints (`10240→70144`) | Longer honest run changes narrative: +0.47% (512) is not stable; longer honest **-6.13%** with high DD/vol shows penalty tuning needed before scaling. `REPORT.md` is stale 3-way table | Report previously 3-way (150k/151k/157k); now 4-way + detailed metrics | Rewrote all §5 sections to `RUN_REPORT.md` as source |
| 11 | Finance | `config.yaml:4` `[q_proj, v_proj]` stale vs `config.py:14` `[wqkv, wo]` auto-remap | YAML vs Python mismatch (same as anomaly) persists | Report previously trusted YAML | Added subtle note §5.1, §5.3, §7.1 |
| 12 | Finance | `data/README.md:1` + `nifty50_summary_statistics.csv` (`EICHERMOT 998k%`) + `artifacts/finance_test/` + `RUN_REPORT.md:318` glossary | Offline data + glossary not previously inventoried | Complete go-through requires | Added to §5.2, §5.5 artifacts, §7.3 |
| 13 | All | `COMBINED_REPORT.md` itself now 1156 lines → will grow ~150 lines; `F:/` root now 13 entries incl `COMBINED_REPORT.md` + `C68-FinanceRL/` + `anomaly gi/` + `deepfake gi/` | Self-reference needed for future audits | Added this §7.5 |

All other files (e.g., `anomaly gi/C68-Anomaly/README.md` 304 lines, `deepfake gi/.../ppt content.txt` empty) unchanged since last revision.

---

## 8. How to Reproduce

### 8.1 Prerequisites (A5000, Windows/Linux)

```bash
# Torch CUDA (check nvidia-smi → CUDA Version, driver 595.95 → cu128)
pip install torch --index-url https://download.pytorch.org/whl/cu128
# or cu121 / cu118 as per nvidia-smi
pip install -r requirements.txt
# transformers ≥4.37, peft 0.14, accelerate 0.33, bitsandbytes 0.45, datasets 3.0, scikit-learn 1.4
```

### 8.2 Anomaly GI (`anomaly gi/C68-Anomaly/README.md:136`, `PROJECT_REPORT.md:568`)

```bash
# From anomaly gi/C68-Anomaly/
# 0. Smoke preprocessing (5 runs/fault, ~10 sec, isolated to _smoke)
python scripts/prepare_tep.py --config configs/config.yaml --smoke --log-level INFO
# Full A5000 (production, scaler on Training only, ~4GB)
python scripts/prepare_tep.py --config configs/config.yaml --full
# or A5000 config
python scripts/prepare_tep.py --config configs/config_a5000.yaml --full

# 1. Detector training (normal-only)
python scripts/train_anomaly_detector.py --config configs/config.yaml          # 50ep batch64
python scripts/train_anomaly_detector.py --config configs/config_a5000.yaml    # A5000 batch128

# 2. Evaluate detector
python scripts/evaluate_anomaly_detector.py --config configs/config.yaml

# 3. LLM dataset (skip if A5000 not in use — code ready)
python scripts/generate_llm_dataset.py --config configs/config.yaml --samples-per-fault 8
# A5000 recommended: --samples-per-fault 20-30

# 4. Adapter training (on A5000 only)
python scripts/train_tep_adapter.py --config configs/config.yaml          # QLoRA 6-8GB
python scripts/train_tep_adapter.py --config configs/config_a5000.yaml    # BF16 12-14GB

# 5. Inference & sweep
python scripts/test_adapter.py --config configs/config.yaml
python scripts/test_end_to_end.py --config configs/config.yaml --inject-at 800
python scripts/test_end_to_end.py --config configs/config.yaml --no-llm --inject-at 800  # fallback
python scripts/evaluate_all_faults_full.py --config configs/config_a5000.yaml
python scripts/diagnose_fault15_full.py --config configs/config.yaml
python scripts/diagnose_faults_3_9_15.py --config configs/config_a5000.yaml

# 6. API/UI
uvicorn api.server:app --host 0.0.0.0 --port 8000
streamlit run ui/app.py
# Follow-up QA example
python -c "from main import TEPApp; from utils import load_config; app=TEPApp(load_config()); print(app.answer_followup('ANOM-0001','Which sensors?'))"
```

### 8.3 Deepfake GI (`deepfake gi/LY_C68/deepfake_project/REPORT_exp2_to_exp3.md:140` + alternate stacks)

```bat
REM Corrective fine-tune from ep010 (one line) — current best (lock scheduler, don't re-stretch)
python train.py --dataset_root "C:\Users\Admin\Desktop\ly project c 68\deepfake_project\datasets\hydrafake" --json_root "C:\Users\Admin\Desktop\ly project c 68\deepfake_project\datasets\hydrafake\jsons" --output_dir ./runs/intern_exp3_corrective --weights_from ./runs/intern_exp2/checkpoints/ep010.pth --epochs_cls 0 --epochs_joint 2 --lr 1e-5 --lm_loss_weight 0.3 --lm_seq_len 512 --batch_size 4 --grad_accum 8 --num_workers 8 --freeze_cls_head --freeze_projector

REM Continue after epoch 1 (keep total_steps 17682, don't add epochs_joint 5)
python train.py --dataset_root "C:\..." --json_root "C:\...\jsons" --output_dir ./runs/intern_exp3_corrective --resume ./runs/intern_exp3_corrective/checkpoints/ep001.pth --epochs_cls 0 --epochs_joint 2 --lr 1e-5 --lm_loss_weight 0.3 --lm_seq_len 512 --batch_size 4 --grad_accum 8 --num_workers 8 --freeze_cls_head --freeze_projector

REM Sweeps already logged (lr5e-6 lm0.12 proj-only) — from ep001.pth
python train.py --dataset_root "C:\..." --json_root "C:\...\jsons" --output_dir ./runs/intern_exp3_corrective --weights_from ./runs/intern_exp3_corrective/checkpoints/ep001.pth --epochs_cls 0 --epochs_joint 2 --lr 5e-06 --lm_loss_weight 0.12 --lm_seq_len 512 --batch_size 4 --grad_accum 8 --num_workers 8 --freeze_projector

REM Evaluate cls-only across 4 splits (trusted ep010)
python eval.py --dataset_root "C:\...\datasets\hydrafake" --json_root "C:\...\datasets\hydrafake\jsons" --model_path ./models/InternVL3-2B --checkpoint ./runs/intern_exp2/checkpoints/ep010.pth --batch_size 8 --num_workers 8
REM expect: ID 0.815/0.900 CM 0.978/0.997 CF 0.779/0.917 CD 0.723/0.796 MEAN 0.824/0.903 (eval_ep010.log:12)

REM Inference deterministic forensic
python inference.py --model_path ./models/InternVL3-2B --checkpoint ./runs/intern_exp3_corrective/checkpoints/best.pth --images 1.jpg 2.jpg 3.jpg 4.jpg --max_new_tokens 512

REM Alternate stacks (for completeness — not InternVL3)
REM RUN2 CLIP+Transformer+GRU (best 0.9045 16ep)
python RUN2/train.py  --help
REM datasets dual-branch CLIP+FFT (exp_final_single best 0.8928)
python datasets/train.py --help
```

### 8.4 Finance RL (`C68-FinanceRL/RUN_REPORT.md:285` supersedes `REPORT.md:208`, `README.md:46`)

```powershell
# From C68-FinanceRL/
# CUDA torch first
pip install torch --index-url https://download.pytorch.org/whl/cu128
# Bundled data already in data/nifty50_prices.csv — verify
dir data
python evaluate_baselines.py --data data\nifty50_prices.csv --start 2023-01-01 --end 2024-12-31

# Smoke honest (~30-70s, proves pipeline + holdout)
python train.py --data data\nifty50_prices.csv --train-end 2022-12-31 --timesteps 512 --output artifacts\smoke_a5000 --device cuda --dtype bf16 --eval-interval 256
python evaluate_policy.py --data data\nifty50_prices.csv --artifacts artifacts\smoke_a5000 --start 2023-01-01 --end 2024-12-31 --device cuda
python evaluate_baselines.py --data data\nifty50_prices.csv --start 2023-01-01 --end 2024-12-31

# Production honest (75k ≈70 episodes ~1.3h; 100k ~94 episodes ~1.8h)
python train.py --data data\nifty50_prices.csv --train-end 2022-12-31 --timesteps 75000 --output artifacts\finance_prod --device cuda --dtype bf16 --rollout-steps 512 --minibatch-size 128 --eval-interval 5000 --checkpoint-interval 10000
python evaluate_policy.py --data data\nifty50_prices.csv --artifacts artifacts\finance_prod --start 2023-01-01 --end 2024-12-31
# Re-evaluate leaky run for comparison
python evaluate_policy.py --data data\nifty50_prices.csv --artifacts artifacts\finance_v1 --start 2023-01-01 --end 2024-12-31

# Optional rebuild data from yfinance
python prepare_nifty50_data.py --start 2020-01-01 --end 2025-01-01 --output data\nifty50_prices.csv
```

### 8.5 Config References

- Anomaly master: `anomaly gi/C68-Anomaly/configs/config.yaml:1` (seed42, window60 stride5, detector 128/2/64 dp0.05 ep100 lr1e-3 p99.5, evidence top5 min3% trend8 onset1.5 max6, events 3/20/200, LLM InternVL2-2B tep_rca r16 bf16 batch2×8 ep3 lr2e-4 seq4096).
- Deepfake defaults: `deepfake gi/LY_C68/deepfake_project/train.py:342` args batch8 grad_accum4 epochs_cls3 epochs_joint2 lm_seq_len512 lr5e-5 lora_rank64 α128.
- Finance defaults: `C68-FinanceRL/config.py:6` ModelConfig backbone `internlm/internlm2_5-1_8b-chat` r16 wqkv/wo bf16; `config.py:21` PPOConfig γ0.99 λ0.95 clip0.2 rollout256 minibatch64 (prod 512/128).

---

## 9. Appendices — File Maps

### A. Anomaly File Map (`anomaly gi/C68-Anomaly/README.md:252`, `PROJECT_REPORT.md:544`)

```
configs/config.yaml, config_a5000.yaml
preprocessing/{tep_loader.py, scaler.py, windowing.py}
anomaly_detection/{lstm_autoencoder.py, dataset.py, train.py, threshold.py, inference.py, lstm_predictor.py}
evidence/{sensor_contribution.py, temporal_analysis.py, process_relationships.py, event_builder.py}
llm/{dataset.py, model.py, train_adapter.py, evaluate.py, adapter_loader.py, inference.py}
streaming/simulator.py
events/event_store.py
main.py: TEPApp
scripts/{prepare_tep.py, train_anomaly_detector.py, evaluate_anomaly_detector.py,
         generate_llm_dataset.py, generate_synthetic_rca_v2.py, train_tep_adapter.py,
         test_adapter.py, test_end_to_end.py, evaluate_all_faults_full.py,
         diagnose_fault15_full.py, diagnose_fault15_stages3_6.py, diagnose_fault15_sensor_reconstruction.py,
         diagnose_faults_3_9_15.py, diagnose_sensor_aware.py, diagnose_relationships.py}
outputs/preprocessing/{scaler.pkl,baseline_stats.json}
outputs/anomaly_detector/{model.pt,threshold.json,config.json,normal_val_scores.json}
outputs/tep_rca_adapter/tep_rca/{adapter_config.json,adapter_model.safetensors}
outputs/evaluation/{fault15_*, all_faults_*, prediction_*, relationship_*, evaluation.json, training_summary.json}
data/processed/{normal_windows_*.npy, fault_values/fault_XX.npy, manifests/*.json}
data/llm/{train,val,test}/
api/server.py
ui/app.py
```

### B. Deepfake File Map (re-audited complete — 100+ files)

```
deepfake_project/
  dataset.py (HydraFakeDataset, build_dataloaders, _resolve_path, _TEST_GENERATOR_MAP:40)
  model.py (DeepfakeReasoningModel:138, CustomProjector:40, ClassificationHead:89, DomainRouter:118)
  train.py (train_epoch:209, validate:300, build_scheduler:199, save_checkpoint atomically:185, weights_from:356, freeze flags:361)
  eval.py (eval_split cls-only, 4 splits ID/CM/CF/CD, setup_logger)
  inference.py (parse_reasoning:98, trim_after_answer:106, format_result:124, build_transform:78)
  datasets/hydrafake/{jsons/{train(144k entries)/val/test/id/cm/cf/cd}, train/val/test/{real,fake}/{generator}/}
  models/InternVL3-2B/
  runs/intern_exp2/{checkpoints/{ep010.pth 5.16GB ZIP_OK, ep011.pth, best.pth 213MB corrupt}, logs/{train.log ~20 resumes, epochs.csv:1 13 rows, steps.csv, curves.png, eval_ep010.log:12 (ID 0.815 CM 0.978...), eval_ep*.log×6, inference_ep*.log×8, inference_logs/inference_*.log×2}}
  runs/intern_exp3_corrective/{checkpoints/{ep001.pth, best.pth}, logs/{train.log:1 187 lines 5 launches, epochs.csv:1 1 row, steps.csv, inference_best_*.log×3 (093031,093538,111437)}}
  RUN1/{dataset.py, model.py:1 (HydraFakeNet scratch 138M), train.py, inference.py, utils.py, transformer_blocks.py, environment.yml, runs/exp1/train.log:144 (loss3.58 acc0.51 valAUC0.54 collapse)}
  RUN2/{model.py:1 (CLIP ViT-B/32 + Transformer 4×512 8 queries + GRU 138M:259, freeze partial), dataset.py, train.py, evaluate.py, inference.py, utils.py, environment.yml, runs/exp1/{train.log:55 best0.9045 16ep, eval/eval.log, training_curves.png}}
  datasets/{model.py:1 (585 lines CLIP 256 + FFTBranch 256 →512 DropPath0.1 depth2 queries4), dataset.py, model.py, train.py, evaluate.py, inference.py, utils.py, test_images.py, runs/{exp3/{train.log 48ep, training_curves.png, eval/eval.log 0.4999 collapse}, exp7/{train.log 59ep AUC0.9644, training_curves.png, eval/eval.log 0.4999}, exp_final_single/{train.log 45ep 0.8928, training_curves.png, eval/eval.log}, exp{5,6,8}/train.log}}
  requirements.txt:1 (torch 2.6.0+cu126 transformers 4.37.2 peft 0.7.1 accelerate 1.14)
  swat_annotations.jsonl, {1,2,3,4,5,6,8,9,10,chick,mar,om}.jpg, run_all_eval.bat
  REPORT_exp2_to_exp3.md:1 144 lines, ppt content.txt:1 empty
```

### C. Finance File Map (`C68-FinanceRL/RUN_REPORT.md:102` supersedes `REPORT.md:105`, `README.md:12` — re-audited)

```
C68-FinanceRL/
  RUN_REPORT.md:1 334 lines (current truth) + RUN_REPORT.md.bak (identical) + REPORT.md:1 232 lines (stale) + README.md:1 94 lines
  config.py:1 (ModelConfig backbone internlm2_5-1_8b-chat wqkv/wo r16 bf16, PPOConfig γ0.99 λ0.95 clip0.2) + config.yaml:1 11 lines stale [q_proj,v_proj] → auto-remapped
  env.py:45 (PortfolioEnv, pivot().ffill().dropna 2017-11-17, window30 10bps 30% cap, reward net-penalties, info equity/gross/net/turnover/cost/dd/vol)
  model.py:80 (InternLMFinancePolicy + FinanceEncoders, wqkv remap, use_cache False hidden_states[-1][:,-1], bf16 cast)
  ppo.py:17 (compute_gae, ppo_update bf16/fp16 branches)
  train.py:36-40 (A5000 loops bf16 TF32 high, autocast, turnover bf16→float fix, logging mean_reward/net/max_dd/turnover/peak, checkpoints checkpoint_<steps>/, holdout --eval-interval)
  evaluate_policy.py:1 (loads finance_lora/finance/adapter_config.json nested + CUDA deterministic mean, saves eval_*.json)
  evaluate_baselines.py:1 (equal-weight 1/49)
  prepare_nifty50_data.py:1 (yfinance .NS)
  data/{nifty50_prices.csv 287k rows 14MB, nifty50_summary_statistics.csv (EICHERMOT 998k% etc.), README.md:1 (Date,Ticker,Close,Volume format)}
  artifacts/{finance_v1 (25k leaky 157k), finance_test, finance_prod (7 checkpoints 10240→70144 + final eval 141k -6.13%), smoke_honest/checkpoint_{256,512} (512→151k +0.47%), .gitkeep}
  tests/README.md + requirements.txt:1 (numpy pandas gymnasium torch≥2.4 transformers≥4.45 peft≥0.13 accelerate yfinance; verified 2.11+cu128)
```

### D. Report Lineage & Provenance

- TEP simulation: Downs & Vogel 1993, Rieth et al. consolidated (fault onset 160).
- LLM fault knowledge: `anomaly gi/C68-Anomaly/scripts/fault_knowledge.py` — supervised labels informed by TEP literature, not per-sample copied; runtime evidence never consults it.
- HydraFake: generator-organized forgery taxonomy (21 generators → 4 domains).
- Nifty 50: Yahoo Finance `.NS` via `prepare_nifty50_data.py` + current NSE list (survivorship bias disclosed).
- Seed discipline: anomaly `seed:42` everywhere (`configs/config.yaml:8`); deepfake WeightedRandomSampler + deterministic greedy; finance PPO std.
- Evaluation honesty: anomaly fault-disjoint 18/2/2 + simulationRun-disjoint; deepfake richest-tag per-image; finance chronological holdout.

---

> **Closing note on what this combined work proves (updated after complete re-audit):** Each domain independently discovered that **the hardest failures are not bugs but indistinguishable-in-representation regimes** (TEP stealth drifts inside noise at `config_a5000.yaml:62` 99.0 vs `threshold.json` 99.5, HydraFake bare 72% vs rich 28% + 0.499 collapse on `datasets/runs/exp3/eval.log`, portfolio churn vs return noise `RUN_REPORT.md:220` gross 0.086% < baseline). The corrective pattern is shared: **freeze what works** (TEP detector frozen, HydraFake `cls_head+projector` 16 tensors `train.log:8`, Finance InternLM frozen), **retrain only the smallest adapter on the cleanest supervision** (TEP `tep_rca` `wqkv/wo/w1/w2/w3` `training_summary.json:9`, Deepfake LoRA r64 73M `train.log:9`, Finance LoRA r16 `wqkv/wo`), **and evaluate at operator granularity** (TEP events not windows, Deepfake tags+answer not AUC alone `train.log:87` `eval_ep010.log:12`, Finance holdout ₹1L equity `RUN_REPORT.md:184` not peak). The newly audited `-6.13%` honest longer run (`artifacts/finance_prod/eval_*.json:4` DD22.44% vol0.014) and the 5-launch corrective log (`runs/intern_exp3_corrective/logs/train.log:1`) plus FFTBranch (`datasets/model.py:68`) prove that **scaling without penalty/scheduler discipline hurts** — next milestone is not more steps but *one-knob* ablations: multi-scale windows for TEP, locked-scheduler epoch2 + FFT for Deepfake, `turnover_penalty 0.001→0.003` for Finance — same A5000, same discipline.

*End of Combined Report — generated 2026-09-02 rev v2 from frozen artifacts without re-execution (re-audited 200+ files, 13 deltas §7.5). For questions, rerun the per-domain evaluate scripts listed in §8 to regenerate the 10k-run, 4-split, and 459-day tables.*
