# C68 — ARCHITECTURES ONLY — Diagram + Explanatory Text
### 3-Domain LoRA Adapters on Shared InternVL Backbone (4th Logical Reasoning — Not Started)
> **File:** `F:/ARCHITECTURES.md` — extracted from `F:/C68_CONTEXT_BOMB.md:80` (367k, 3721 lines) and source code with `file:line` citations. No results, no training logs — only architecture.

---

## 0. SHARED BACKBONE — ONE FROZEN TRANSFORMER, THREE NAMED LoRA ADAPTERS

> **Vision:** Single `transformers` + `peft` deployment where `internlm/internlm2_5-1_8b-chat` / `OpenGVLab/InternVL2-2B` / `InternVL3-2B` are *frozen*, and each domain contributes a tiny LoRA that is `save_pretrained` / `load_adapter` swappable without retraining the core. 4th `logic` will reuse the same hook.

```
                    ┌─────────────────────────────────────────────────────┐
                    │   FROZEN BACKBONE FAMILY (bf16, trust_remote_code)  │
                    │                                                     │
                    │  Anomaly: OpenGVLab/InternVL2-2B (InternLM2-Chat-  │
                    │           1.8B backbone + ViT)                      │
                    │  Deepfake: InternVL3-2B (InternViT 448px + Qwen2)   │
                    │  Finance:  internlm/internlm2_5-1_8b-chat           │
                    │  (all Ampere-native sm_86, InternLM2-Chat-1.8B/     │
                    │   Qwen2 — share wqkv/wo fused-attn pattern)        │
                    │                                                     │
                    │  Loaded via: AutoModel.from_pretrained(             │
                    │    "…", trust_remote_code=True,                     │
                    │    torch_dtype=bfloat16)  // anomaly llm/model.py:142,
                    │                           // deepfake model.py:152,
                    │                           // finance model.py:30     │
                    └────────────────────┬────────────────────────────────┘
                                         │ for p in backbone.parameters(): requires_grad=False
                                         │ finance model.py:35, deepfake model.py:190, anomaly llm/model.py:212
                    ┌────────────────────┴────────────────────────────────┐
                    │  TRAINABLE SHIMS (<4% params, all that learns)      │
                    │                                                     │
                    │  TEP `tep_rca`      : LoRA r16 α32 dropout0.05      │
                    │   targets ["wqkv","wo","w1","w2","w3"]              │
                    │   (actually trained training_summary.json:9)        │
                    │   vs doc ["q/k/v/o/gate/up/down"] config.yaml:149   │
                    │   size ~62 MB adapter_model.safetensors             │
                    │                                                     │
                    │  Deepfake reasoning : LoRA r64 α128 dropout0.05      │
                    │   targets ["q_proj","k_proj","v_proj","o_proj",      │
                    │            "gate_proj","up_proj","down_proj"]       │
                    │   model.py:172 — 73,990,788 trainable (3.41%)       │
                    │   logs/train.log:9 LoRA392 tensors                │
                    │                                                     │
                    │  Finance `finance`  : LoRA r16 α32 dropout0.05      │
                    │   targets ["wqkv","wo"] (fused, auto-remapped       │
                    │   from stale [q_proj,v_proj] config.yaml:4          │
                    │   via model.py:39-53 leaf detection)                │
                    │   finance_lora/finance/adapter_config.json          │
                    │                                                     │
                    │  Plus domain heads: FinanceEncoders 3×MLP,          │
                    │   Actor 2048→49 Critic 2048→1 log_std -0.5          │
                    │   finance model.py:12-15 / 71-75                    │
                    │  Deepfake heads: CustomProjector 2 tokens +         │
                    │   ClassificationHead vision_dim*2→256→2               │
                    │   model.py:40-84 / 89-113                           │
                    │                                                     │
                    │  All created via:                                   │
                    │   LoraConfig(r,alpha,dropout,target_modules,       │
                    │              bias="none", task_type=FEATURE_         │
                    │              EXTRACTION) → get_peft_model(          │
                    │     backbone, lora_config, adapter_name="finance"|  │
                    │     "tep_rca"|"deepfake")  finance model.py:56-65   │
                    └─────────────────────────────────────────────────────┘
                                         │
                    ┌────────────────────┴────────────────────────────────┐
                    │  DOMAIN ENCODERS → INPUTS_EMBEDS → LLM → HEADS      │
                    │  (each domain translates its modality into token     │
                    │   embeddings the frozen LLM can read)               │
                    └─────────────────────────────────────────────────────┘
```

**Why this shape:** InternVL/InternLM already understand long sequences. Instead of training a 1.8–2B transformer from scratch on 44k windows / 35k traces / 2k days, we freeze the 24-layer hidden 2048 prior and only train <80M parameters that map domain signals into its token space. `model.py:190-192` vision frozen, `model.py:35` backbone frozen, optimizer built *after* freezes `train.py:437-454`. Adapter is portable: `backbone.save_pretrained(output/finance_lora)` + `torch.save(encoders, actor, critic, log_std → ppo_heads.pt)` `finance model.py:104-108` — can be `PeftModel.from_pretrained` or `load_adapter` on any host without touching base weights. Same pattern will host `logic` — new `LoraConfig(adapter_name="logic")` sharing the same `AutoModel`.

---

## 1. ANOMALY GI — TEP INDUSTRIAL ANOMALY DETECTION & AUTO-RCA

### 1.1 One-Line Idea (PROJECT_REPORT.md:4)

*52 sensors streaming from a Dow Chemical Tennessee Eastman Process simulation. A sensor brain that has *never seen a fault* learns to reconstruct "normal" and flags reconstruction failure. A separate language brain (InternVL2-2B + one LoRA `tep_rca`) reads the sensor brain's evidence JSON and writes an auditable incident report. The LLM has never seen raw sensors.*

**Design principles** `PROJECT_REPORT.md:45-49` + `utils.py:24-106` `_DEFAULT_CONFIG`:

1. No fault labels for detection — must fire on unknown future fault
2. Explainability by construction — every claim traced to `% deviation / z / trend / onset order`, disclaimer `correlation ≠ causation` `process_relationships.py:144` `event_builder.py:214`
3. Separation — LoRA holds *only* language reasoning; scaler/AE/threshold/evidence/streaming outside → detector swappable without LLM retrain, and vice versa
4. Production streaming — `TEPApp.process_sensor_stream(record)` one record at a time, event aggregation (no LLM per timestamp), SQLite `events/event_store.py:22`, FastAPI `api/server.py:97`, Streamlit `ui/app.py:140`

### 1.2 Two-Brain Block Diagram (PROJECT_REPORT.md:99-125 + README.md:28-34)

```
┌──────────────────────────────────────────────────────────────────────────────┐
│  LAYER A — SENSOR / TIME-SERIES PIPELINE (no LLM)                            │
│  preprocessing/  →  anomaly_detection/  →  evidence/  →  events/             │
│  streaming/ (simulator)  •  all math lives here, zero LLM weights            │
│                                                                              │
│  LAYER B — LANGUAGE REASONING (frozen LLM + one adapter)                     │
│  llm/  (OpenGVLab/InternVL2-2B = InternLM2-Chat-1.8B backbone) + tep_rca     │
│        QLoRA 4-bit NF4 double-quant bf16 (6-8GB) or BF16 LoRA (12-14GB)      │
│        r16 α32 dropout0.05 targets wqkv/wo/w1/w2/w3 (fused)                  │
│                                                                              │
│  ORCHESTRATION: main.py:72 TEPApp  •  INTERFACES: api/server.py + ui/app.py  │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 1.3 End-to-End Data Flow (main.py:122-346 + evidence/event_builder.py:113-241)

```
sensor_record {values:[52] float32, sample_index:int, fault_label?:int}
  │  main.py:122 process_sensor_stream(record) validates 1D 52
  ↓  deque buffer 60  main.py:107  (holds 60 most recent)
  │  emit every stride 5  main.py:152  (so 60-window step 5)
  ↓  window [60,52]  float32  windowing.py:19 to_windows()
  │  scaler.transform()  scaler.py:163 load_scaler() → StandardScaler fitted ONLY on
  │                      data/raw/normal/TEP_FaultFree_Training.csv 250k rows (Rieth)
  │                      preprocessing/scaler.py:97 + prepare_tep.py:75  •  no fault leakage
  ↓  scaled window [60,52]
  │  LSTMAutoencoder  anomaly_detection/lstm_autoencoder.py:101
  │    Encoder: LSTM(52 → 128 ×2 layers, dropout0.05, bidirectional false) → latent64
  │             via Linear(256→64)  lstm_autoencoder.py:28-54  (utils.py:53 default 32/16 overridden by config.yaml:69-73)
  │    Decoder: latent64 → Linear→h0/c0 + repeat latent per-step → LSTM(64→128×2) → Linear→52
  │             lstm_autoencoder.py:57-99
  │    Training: ONLY normal windows, MSE loss train.py:25 _mse_loss, AdamW lr1e-3 cosine
  │              train.py:109, epochs100 config.yaml:76 (default 50 utils.py:55), batch64 (A5000 128)
  │              earlystop15 grad1.0 ckpt5 → outputs/anomaly_detector/model.pt 504k params
  │    Inference: inference.py:94-133 score_window(window) → score=mean(per_sensor MSE [52])
  │              + per_sensor vector [52]; score_windows batched 256; is_anomalous(score>thr)
  ↓  scalar score (global MSE) + per_sensor[52]
  │  threshold = 0.6873631477355957  data/processed/anomaly_detector_eval.json:2
  │          = p99.5 of 6,675 normal val reconstruction errors mean0.619 std0.028 n6675
  │          threshold.py:40 compute_threshold(method=percentile)  •  config.yaml:89 99.5
  │          vs config_a5000.yaml:62 99.0 (inversion — doc says less aggressive but file more aggressive)
  ↓  recent_flags deque(maxlen = max_event_windows)  main.py:188
  │  _feed_aggregator  main.py:182-219: 3 consecutive anomalous → open event
  │                              20 normal → close event (or cap 200)  main.py:207-218
  │                              NO LLM call per window
  ↓  raw event windows [N,60,52] (open→close)
  │  build_event()  evidence/event_builder.py:113-241
  │    • per-sensor error → contribution% + deviation% (current-baseline)/baseline*100 scaler.py:61
  │      + z=(current-baseline)/std scaler.py:77 sensor_contribution.py:82/106
  │    • sensor_trend() linear fit window8 → inc/dec/stable temporal_analysis.py:18
  │    • detect_onsets() error>1.5×std temporal_analysis.py:40 + analyze_temporal_sequence() sorted minutes
  │    • suggest_subsystems() maps top5 deviating → candidate_subsystem + score process_relationships.py:173
  │      from 52 SENSOR_NAMES XMEAS_1..41 + XMV_42..52 + 8 SUBSYSTEM_GROUPS + 21 KNOWN_PROCESS_RELATIONSHIPS
  │    • severity = score/thr ratio ≥3.0 critical ≥1.8 high ≥1.2 medium else low event_builder.py:102
  │    • always "temporal ordering does not prove causation" event_builder.py:192
  ↓  EventEvidence {
       anomaly_score:float, top5 sensors [{name, deviation%, z, trend, contribution%}],
       temporal_sequence: [{sensor, onset_min} up to 6], candidate_subsystem, uncertainty disclaimer,
       pre_post_context baseline_windows20
     }  config.yaml:97-104 top5 min3.0 baseline20 trend8 onset1.5 max6
  │  main.py:223-241 build_event(scaler baseline, threshold, config)
  ↓  EventEvidence → JSON (evidence JSON, never raw windows)
  │  either:
  │    A) LLM ONLINE: llm/inference.py:206 RCAInference.generate_report(evidence_dict)
  │       InternVL2-2B + tep_rca  llm/inference.py:159 dummy pixel_values 448 + image_flags0
  │       GenerationMixin patched llm/model.py:212 manual greedy decode max_new_tokens768 do_sample false temp0.7
  │       prompt = system + evidence text (anomaly_score, sensor deviations, trends, temporal order,
  │                candidate subsystem, context) → JSON {summary, root_cause, subsystem, evidence,
  │                reasoning, severity, confidence, action, uncertainty}
  │    B) LLM OFFLINE: main.py:34 _fallback_report() deterministic, still grounded in same evidence
  ↓  JSON report
  │  events/event_store.py:95 store_event() → SQLite outputs/events.db
  │    tables events + followups  event_store.py:22 next_event_id→ANOM-0001 event_store.py:82
  │    conversation_history() event_store.py:177 for follow-up context
  ↓  operator UI
     Streamlit ui/app.py:140 tabs live/stream/events/chat threshold sidebar
     or curl: main.py:257 answer_followup(event_id, question) →
            llm/inference.py:229 answer_followup(event, report, history, question) with stored evidence+report
```

**Why this shape:** TEP faults are *temporal* (60 samples ≈ 12% of 500-sample run, fault at sample 160 `config.yaml:39`). `windowing.py:44` leak-free split by `simulationRun` (whole segments, not random windows) prevents temporal leakage. `scaler.py:98` normal-only avoids fault leakage. `lstm_autoencoder.py:142` high capacity (128h latent64) reconstructs normal perfectly — but that is the bug for stealth drafts (see §1.6). Evidence bridge forces LLM to ground: it never sees `[52]`; only `deviation% + z + trend`.

### 1.4 Module-to-File Map

| Block | File | Key line |
|-------|------|----------|
| TEP loader (Rieth vs 52-col auto-detect, chunksize 500k) | `preprocessing/tep_loader.py` | `tep_loader.py:56-60` format, `110` missing, `164` stream |
| Scaler/Baseline (StandardScaler, percent/z) | `preprocessing/scaler.py` | `scaler.py:97` build(), `61` dev, `77` z, `163` load_scaler |
| Windowing (60×52, float32, segment split) | `preprocessing/windowing.py` | `windowing.py:19` to_windows, `44` segment_ids, `57` split |
| LSTM-AE train (MSE, AdamW cosine, early15) | `anomaly_detection/train.py` | `train.py:25` _mse_loss, `109` cosine, `191` save |
| Threshold (percentile p99.5 / mean_std) | `anomaly_detection/threshold.py` | `threshold.py:40` compute_threshold |
| Inference (score_window mean(per_sensor)) | `anomaly_detection/inference.py` | `inference.py:94` score_window |
| Alternative predictor (rejected, pred<recon) | `anomaly_detection/lstm_predictor.py` | `lstm_predictor.py:1` |
| Sensor contribution (top5, dev%, z) | `evidence/sensor_contribution.py` | `sensor_contribution.py:38` top_k, `82` dev |
| Temporal (onset, trend, pre/post) | `evidence/temporal_analysis.py` | `temporal_analysis.py:18` trend, `40` onsets |
| Relationships (52 names, 8 subsystems, 21 relations) | `evidence/process_relationships.py` | `process_relationships.py:19` names, `74` groups, `119` relations |
| Event builder (severity ratio, disclaimer) | `evidence/event_builder.py` | `event_builder.py:102` severity |
| Stream simulator (SensorStream replay) | `streaming/simulator.py` | `simulator.py:31` |
| TEPApp orchestration (deque60, 3/20/200) | `main.py` | `main.py:107` deque, `182` _feed_aggregator, `122` process |
| EventStore (SQLite ANOM-0001) | `events/event_store.py` | `event_store.py:22` schema |
| InternVL2-2B LoRA | `llm/model.py` | `llm/model.py:27` INTERNLM2_LORA_TARGETS `["wqkv","wo","w1","w2","w3"]` |
| Adapter training (QLoRA/BF16, 3ep) | `llm/train_adapter.py` | `llm/train_adapter.py:125` merge conflict, `212` checkpoint |
| Inference (dummy 448, JSON) | `llm/inference.py` | `llm/inference.py:159` dummy, `206` generate_report |
| Config master | `configs/config.yaml` | `configs/config.yaml:1-203` |
| A5000 profile | `configs/config_a5000.yaml` | `configs/config_a5000.yaml:1-146` |

### 1.5 Config Snapshot (utils.py:24 + configs/config.yaml:203 merged)

```
seed:42 log_level:INFO
dataset: {format tep_rieth_csv, num_features52, fault_onset160, missing interpolate, clip99.9}
preprocessing: {scaler standard}
windowing: {window60, stride5, float32, block500, test_ratio0.2}
anomaly_detector: {model_dir outputs/anomaly_detector, lstm {hidden128 layers2 latent64 dropout0.05 bidirectional false}, train {epochs100 batch64(A5000 128) lr1e-3 weight1e-5 cosine val0.15 early15 grad1.0 ckpt5}, threshold {method percentile percentile99.5(a5000 99.0) k_std6.0}}
evidence: {top5 min3.0 baseline20 trend8 onset1.5 max6}
events: {db outputs/events.db, confirm3 separate20 max200}
streaming: {window60 stride5 replay0.0 inject null}
llm: {base OpenGVLab/InternVL2-2B, adapter tep_rca, adapter_dir outputs/tep_rca_adapter, dummy448, lora {r16 α32 dropout0.05 bias none targets wqkv/wo/w1/w2/w3}, training {use_4bit true(a5000 false) bf16 true batch2(A5000 4) grad_accum8(A5000 4) epochs3 lr2e-4 weight0.01 max_seq4096 warmup0.03 cosine max_grad1.0 ckpt true logging10 save200 eval200 workers2(A5000 4) train_faults[1,2,3,5,6,7,8,9,10,11,12,13,16,17,18,19,20,22] val[4,14] test[15,21]}, generation {max768 do_sample false temp0.7 top_p0.9 top_k50 rep1.0}}
```

---

## 2. DEEPFAKE GI — HYDRAFAKE DETECTION + REASONING

### 2.1 One-Line Idea (REPORT_exp2_to_exp3.md:13)

*Classify a facial image as `real` vs `fake` **and** generate a structured forensic chain-of-thought `<fast>→<planning>→<reasoning>→<reflection>→<conclusion>→<answer>real|fake</answer>` grounded in visual evidence. InternViT sees pixels, Qwen2 speaks reasoning, LoRA teaches the language.*

### 2.2 Evolution — Three Generations Before Current (why architecture changed)

```
GENERATION 1 — RUN1 HydraFakeNet (from-scratch ViT, no pretraining) — 60.8M
  RUN1/model.py:220 ViTEncoder (image224 patch16 embed512 depth8 heads8)
       ↓
  RUN1/model.py:21  ClassificationHead (LayerNorm512 → 512→256→2 on CLS only)
       ↓                      ↘
  RUN1/model.py:46  ReasoningDecoder (GRU 512 hidden ×2 layers max128 teacher-forced)
  Result: RUN1/runs/exp1/train.log:144 acc0.51 AUC0.543 random — no prior, insufficient data
       → abandoned

GENERATION 2 — RUN2 CLIP ViT-B/32 Partial Freeze (138M) — pretrained pixel prior
  RUN2/model.py:292 CLIPVisionModel openai/clip-vit-base-patch32 CLIP_DIM768
       ↓  freeze logic RUN2/model.py:317 all/partial(last4+post_layernorm)/none default partial
  RUN2/model.py:23  FeatureProjection 768→1024→512 + LayerNorm
       ↓
  RUN2/model.py:57  ScratchMHSA/FFN/Block (qkv 512→1536 scaled dot) 4 blocks
       ↓  expand 512→8×512 + query_tokens 8
  RUN2/model.py:130 ClassificationHead (mean over 8 queries → 512→256→2)
       ↓                      ↘
  RUN2/model.py:179 ReasoningDecoder (GRU 512×2 max128)
  Params: Total138,579,260 Trainable79,476,284 RUN2/runs/exp1/train.log:25
  Result: RUN2/runs/exp1/train.log:55 ep8 Best ValAcc0.9045 AUC0.966 — but eval 0.4999 fake-only collapse
       → pixel-only still GAN-specific, not generalizable cross-generator

GENERATION 3 — datasets/model.py:585 Dual-Branch CLIP+FFT (131M) — frequency prior
  ┌─ Branch1: CLIP ViT 768 → FeatureProjection256 (768→512→256) datasets/model.py:146
  │                ↓
  │         ┌──────┴──────┐
  │    ┌────▼─────┐  ┌────▼─────┐  WHY: GAN upsampling grid spikes vs real camera
  │    │ CLIP 256 │  │ FFT 256  │  frequency distribution; FFT magnitude spectrum
  │    └────┬─────┘  └────┬─────┘  captures spectral fingerprints more universal
  │         └──────┬──────┘      datasets/model.py:70-84 comment
  │              concat 512  datasets/model.py:536 fused = clip256 concat fft256
  │                ↓         assert model_dim == CLIP_PROJ+FFT_OUT datasets/model.py:444
  │  FFTBranch datasets/model.py:69: 2D FFT fft2 ortho → abs → log1p → fftshift
  │     → conv 3→32 BN GELU Pool224→112 →64 BN GELU Pool→56 →128 BN GELU AdaptiveAvg4→4×4
  │     → Flatten2048 → Linear512→GELU→Dropout0.3→256 LayerNorm
  │                ↓
  │  ClassificationHead datasets/model.py:264 depth2 queries4 DropPath0.1 dropout0.4
  │     (down from 4/8 dropout0.2 — less capacity to reduce overfitting)
  │  ScratchTransformerBlock with DropPath datasets/model.py:238 stochastic depth
  │  linearly increasing dpr 0→0.1 datasets/model.py:290
  │                ↓
  │  ReasoningDecoder inference-only datasets/model.py:325 (never receives gradients,
  │     forward ignored datasets/model.py:547 — prevents GRU corrupting classifier)
  │
  │  Params: Total131,708,508 Trainable72,605,532 datasets/runs/exp7/train.log:17
  │  Result: exp_final_single 0.8928 ep17 (clean) but exp3 eval 0.4999 fake-only ×5 repeats
  │          exp7 Best AUC0.9644 ep29 but eval 0.4999 collapse — domain shift + test leak (merged test into train datasets/dataset.py:292)
  │       → still collapsed on cross-domain CD 0.723

CURRENT — InternVL3-2B + LoRA (2.17B total, 73M trainable 3.41%) — foundation prior
  model.py:152 AutoModel InternVL3-2B torch_dtype=bfloat16
       ↓  LoRA r64 α128 dropout0.05 target [q_proj,k_proj,v_proj,o_proj,gate,up,down] model.py:172 BEFORE freeze (fix #1)
  model.py:190 vision_model frozen, model.py:211 assert LoRA trainable → LoRA392 tensors
       ↓  dataset.py:105 _resolve_path + dataset.py:179 keep-richest (rich=tagged?) + dataset.py:241 EOS <|im_end|> + Drop>512
  model.py:40 CustomProjector CLS+mean-patch 2 tokens (was 1) vs old — 1024→2048 each GELU LayerNorm (B,2,2048)
  model.py:89 ClassificationHead vision_dim*2→256→2 on CLS concat patch_mean (richer)
  model.py:118 DomainRouter stub (4 domains) kept for ckpt compat, not called fix #8
  Result: runs/intern_exp2 11 epochs 0.864→0.946 AUC, trusted ep010 0.9429 Acc0.858; eval_ep010.log:12
          ID 13819 0.815/0.900 CM 11249 0.978/0.997 CF 12736 0.779/0.917 CD 15468 0.723/0.796 MEAN 0.824/0.903
          Intern_exp3_corrective freeze 16 tensors (4 cls_head +12 projector) Trainable73,990,788 → only LoRA learns
```

**Why each change:** RUN1 no pretraining → random 0.51; RUN2 adds CLIP pixel prior → 0.90 but still GAN-specific overfits to training generators (test collapse 0.50); datasets adds FFT frequency prior + DropPath + less capacity + Mixup → best 0.96 AUC but still 0.499 on unseen domain (spectral helps but not enough); InternVL3 adds 2B foundation sequence prior → 0.94 cross-model 0.997 (vs 0.50 before) — foundation generalizes.

### 2.3 Current InternVL3-2B LoRA Block Diagram (dataset.py:369 + model.py:393 + train.py:594 + inference.py:275)

```
Image file (hydrafake/train/real/...png  or  hydrafake/test/deepfacelab/→test/cd/deepfacelab...)
  │  dataset.py:105 _resolve_path (strips hydrafake/, inserts cd/cm/cf/id via _TEST_GENERATOR_MAP 26 entries)
  │                 if not exists → Missing(skipped) not black fallback (fix P5) dataset.py:300
  ↓  resolved path (normcase deduplicated)
  │  HydraFakeDataset._load two-phase keep-richest dataset.py:179-271
  │    Phase1 candidates[ normcase(path) ] = (rich, resolved, label, raw, ftype)
  │           rich = 1 if _TAG_RE.search(raw) else 0  (_TAG_RE = <fast|planning|reasoning|reflection|conclusion>)
  │           later rich wins (tagged sft/mipo beats bare all/nested/pgrpo) dupe_count 96,583
  │    Phase2 if not rich and train → no_trace 11,277 dropped; val/test fallback raw=""→sanitize
  │           _sanitize_reasoning <20 chars → `<fast>…</fast> <answer>real/fake</answer>` dataset.py:89
  │           target_text = reasoning + tokenizer.eos_token (<|im_end|> id151645) dataset.py:241
  │           if len(tokenizer(target_text))>512 → too_long 1,676 dropped (never truncated mid-answer)
  │           enc = tokenizer(target_text, max_length512 padding max_length) pre-tokenized at load (fix #4)
  │           → sample {image_path, label, reasoning_ids[512], attention_mask[512], reasoning_text, forgery_type}
  │           result: 35,367 usable /48,320 unique / Missing0 / Dups96,583 / No-trace11,277 / Too-long1,676
  │                   100% have <answer>+<fast> mean259 tok 0 truncated last-token==eos REPORT_exp2_to_exp3.md:65
  │           token stats: p50 236 mean272 p90 457 p95 507 p99 576 max746 REPORT_exp2_to_exp3.md:96
  │           old 192: >256 14,798 (40% lose answer) new 512: 0 truncated 3.7× cost 0.76→2.77s/step
  ↓  DataLoader build_dataloaders dataset.py:324
     tokenizer.pad_token = eos_token if None; train WeightedRandomSampler balanced; pin_memory drop_last
  │  batch {pixel_values (B,3,448,448), labels (B), reasoning_ids (B,512), attention_mask (B,512)}
  ↓  Transforms dataset.py:118 train Resize(480)+RandomCrop448+HFlip0.5+VFlip0.1+ColorJitter0.4/0.4/0.2/0.1+Grayscale0.1+GaussianBlur3 0.3+Rotation10 0.3+ToTensor+Normalize [0.485,0.456,0.406]/[0.229,0.224,0.225]+RandomErasing0.2 | val/test Resize448 CenterCrop
  │  Image tensor (B,3,448,448)
  ↓  DeepfakeReasoningModel model.py:137
  │  ├─ Step1: backbone = AutoModel.from_pretrained("InternVL3-2B", trust_remote_code, torch_dtype=bfloat16) model.py:152
  │  ├─ Step2: LoRA BEFORE freeze (fix #1 order) model.py:172 LoraConfig r64 α128 dropout0.05 target [q,k,v,o,gate,up,down] → get_peft_model(language_model)
  │  ├─ Step3: freeze if "lora_" not in name → requires_grad False + vision_model frozen model.py:185
  │  ├─ Step4: dims vision_dim1024 llm_dim2048 model.py:194
  │  ├─ Step5: heads CustomProjector(1024→2048) model.py:40, ClassificationHead(2048→256→2) model.py:89, DomainRouter(4) model.py:118 (stub)
  │  └─ Step6: _assert_lora_trainable model.py:211 raises FATAL if 0, prints "LoRA trainable: 73,990,788 [OK]" model.py:228 cp1252 [OK] fix
  ↓  Forward train.py:210 train_epoch reads reasoning_ids directly (no tokenizer call fix #4)
     _get_vision_features model.py:239 vision_model(pixel_values.to(bfloat16)) → hs (B,N+1,1024) → cls hs[:,0] (B,1024) + patch_mean hs[:,1:].mean (B,1024)
     ├─ Vis for CLS: vis_for_cls = cat(cls,patch_mean, dim -1) (B,2048) → cls_head → cls_logits (B,2) model.py:280-283
     │     cls_loss = CrossEntropy(cls_logits, labels, label_smoothing0.1) model.py:290
     ├─ Vis for LLM: vis_prefix = custom_projector(cls,patch_mean) → (B,2,2048) model.py:296
     │              token_embeds = language_model.get_input_embeddings()(reasoning_ids) (B,512,2048)
     │              inputs_embeds = cat([vis_prefix, token_embeds], dim1) → (B,514,2048)
     │              lm_labels = reasoning_ids clone; lm_labels[attention_mask==0] = -100 (fix #2 via mask not id==0) model.py:304
     │              prefix_ignore (B,2)-100 concat → lm_labels (B,514); prefix_mask ones(B,2) concat full_mask (B,514)
     │              lm_out = language_model(inputs_embeds, attention_mask=full_mask, labels=lm_labels)
     │              lm_loss = lm_out.loss; loss = cls_loss + 0.3*lm_loss (lm_loss_weight0.3 fix #9) model.py:326
     └─ return {cls_logits, loss, cls_loss, lm_loss}
  │  backward: loss.backward() NO GradScaler (bf16) model.py:243 fix P6 → clip_grad_norm 1.0 → optimizer.step → scheduler.step → zero_grad model.py:250
  │  optimizer groups train.py:457 LoRA392 tensors lr=lr (1e-5 corrective vs 5e-5 original) + custom4 dormant (domain_router)
  │  scheduler train.py:199 LambdaLR warmup5% + cosine train.py:476 total_steps = (epochs_cls+epochs_joint)*steps_per_ep (old patchwork 90560 for 11ep, corrective clean 17682 =2*8841)
  ↓  Validation train.py:301 validate cls-only model(pv,labels) → metrics accuracy/f1/auc; train_epoch logs Loss Cls LM per 200 steps logger→ train.log + steps.csv + epochs.csv (13 cols) + curves.png train.py:118 matplotlib
  ↓  Checkpoint train.py:560 save_checkpoint atomic os.replace .tmp + best.pth if cur_auc>best_auc (but careful P10 — pick by LM loss not AUC)
```

**Inference** `inference.py:275` `model.py:338`:

```
Image (1.jpg) → transform 448 Resize CenterCrop ToTensor Normalize → pv (1,3,448,448) → model.encode → cls_logits → softmax → pred {REAL|FAKE} conf%
                                                                    ↓
                                                              vis_prefix (1,2,2048) bf16 → attention_mask ones(1,2)
                                                                    ↓
                                                              language_model.generate(
                                                                inputs_embeds=vis_prefix, attention_mask,
                                                                max_new_tokens512, do_sample=False temp1.0 top_p1.0 rep1.05 use_cacheTrue
                                                                pad_token_id eos, eos_token_id eos)  // deterministic greedy fix P7
                                                                    ↓
                                                              tokenizer.decode → raw_text → trim_after_answer cuts after last </answer> inference.py:106
                                                                             strips <|im_end|> <|endoftext|> → maybe "" if pre-EOS ramble
                                                                    ↓
                                                              parse_reasoning tags + answer unknown fallback inference.py:90
                                                                    ↓
                                                              format_result ordered fast→planning→reasoning→reflection→conclusion → [Raw LM output] if no tags
                                                                 + WARNING if gen answer disagrees with classifier inference.py:124
```

**Prompts** `model.py:24` SYSTEM: `You are image authenticity expert. Firstly <fast>, then <planning><reasoning><reflection><conclusion> as needed, finally <answer>real|fake</answer>`. USER: `Please determine...`.

### 2.4 HydraFake Dataset & Transforms

- On-disk `datasets/hydrafake/{train,val,test}` + `jsons/{train,val,test,id,cm,cf,cd}` — jsons `all.json 48,320 + 26 nested 48,320 + pgrpo_8k 8,033 = bare 104,673 72%` vs `sft_36k 36,750 + mipo_3k 3,480 = rich 40,230 28%` total144,903 for48,320 unique `REPORT_exp2_to_exp3.md:23` val8k/4k unique. `_TEST_GENERATOR_MAP` 26 entries `dataset.py:40` inserts category for test paths `hydrafake/test/deepfacelab→test/cd/...`.

- Strong aug `dataset.py:118` train vs deterministic val/test; label 0 real 1 fake `forgery_type` unknown; `max_text_len512` image448 `WeightedRandomSampler` balanced `pin_memory`.

### 2.5 Config Snapshot (train.py:333 args)

```
batch8 grad_accum4 eff_batch32 lm_seq_len512 lm_loss_weight0.3 epochs_cls3 epochs_joint2 (corrective 0+2) lr5e-5 (corrective 1e-5) lora_rank64 α128 warmup0.05 patience5
```

---

## 3. FINANCE RL — INTERNLM2.5 + PPO PORTFOLIO

### 3.1 One-Line Idea (C68-FinanceRL/REPORT.md:6)

*Frozen language model that already understands sequences manages a basket of 49 Nifty 50 stocks the way a human fund manager would — deciding every day how much to allocate to each asset — learning purely from historical prices via trial and error (PPO), never live market.*

### 3.2 System Overview (REPORT.md:85 + model.py:108 + env.py:87)

```
data/nifty50_prices.csv  (14 MB, 287,263 rows long-format Date,Ticker,Close,Volume  data/README.md:8
                          Raw 1999-01-01→2026-01-30 → effective 2017-11-17→2026-01-30 after pivot().ffill().dropna() env.py:45
                          → 2,027 trading days (train.py logs dates=2027, honest 1266 after train-end 2022-12-31)
                          49 tickers ADANIENT…WIPRO environment_config.json:9, current NSE list → survivorship bias)
         │
         ↓  PortfolioEnv (window30, transaction_cost_bps10.0 max_weight0.30 turnover_penalty0.001
         │               volatility_penalty0.10 drawdown_penalty0.20) env.py:9 FinanceRewardConfig
         │        action_space Box(-inf,inf) shape(49,)  obs space Dict{market Box(window,49,5), portfolio Box(50), risk Box(3)}
         │        reset index=window weights zeros cash1.0 equity100k peak100k env.py:50
         │        step(action): target=_target_weights(action) env.py:56
         │                     turnover=|target-weights|.sum() env.py:57
         │                     cost=turnover*10/10000 env.py:58
         │                     gross=dot(target, prices[index+1]/prices[index]-1) env.py:59
         │                     net=gross-cost equity*=1+net peak=max drawdown=1-equity/peak env.py:60-61
         │                     returns.append(net) vol=std returns[-20:] env.py:62
         │                     reward=net -0.001*turnover -0.10*vol -0.20*drawdown env.py:63
         │                     info {equity,gross,net,turnover, cost, drawdown, vol} env.py:66
         │        _observation env.py:80:
         │          prices window → returns, log_returns, rolling_vol (std 20), ma_gap, volume_z → market stack (window,49,5) env.py:81-85
         │          risk array [std20, 1-equity/peak, equity/initial-1] env.py:86
         │          portfolio = r_[weights, cash_weight] env.py:87
         │
         ↓  {market (30,49,5), portfolio (50,), risk (3,)}
         │  FinanceEncoders model.py:12 hidden2048 n_assets49
         │    Market: Sequential Linear(5→2048)→GELU→Linear(2048)  market→mean(1,2) → (2048)
         │    Portfolio: Sequential LayerNorm(50)→Linear(50→2048)→GELU
         │    Risk: Sequential LayerNorm(3)→Linear(3→2048)→GELU
         │    forward stack dim1 → (B,3,2048) 3 tokens model.py:16
         │
         ↓  (3 tokens ×2048) as inputs_embeds
         │  InternLMFinancePolicy model.py:20
         │    AutoModel internlm/internlm2_5-1_8b-chat 24 layers hidden2048 bf16 trust_remote_code model.py:21-30
         │    for p in backbone.parameters: requires_grad=False model.py:35 freeze
         │    Auto-correct target wqkv/wo model.py:36-54 leaf detection wqkv in Linear leaves → remap q/k/v→wqkv o→wo (stale config.yaml:4 [q_proj,v_proj])
         │    LoraConfig r16 α32 dropout0.05 target ["wqkv","wo"] adapter_name finance bias none TASK_FEATURE_EXTRACTION model.py:56 → get_peft_model
         │    hidden = base.config.hidden_size 2048 model.py:66 keep encoders/actor/critic same dtype as backbone on CUDA
         │    encoders 3×MLP + actor Linear2048→49 + critic Linear2048→1 + log_std Parameter(49, -0.5) model.py:71-75
         │    forward obs→ cast inputs to encoder dtype model.py:79-82 → backbone(inputs_embeds=encoders(market,portfolio,risk), return_dict, use_cache=False, output_hidden_states=True) model.py:85-88
         │              if hidden_states → state=hidden_states[-1][:,-1] elif last_hidden_state else error model.py:89-95
         │              actor(state) → mean (B,49), critic(state).squeeze→ value (B,)
         │    _dist: Normal(mean, log_std.exp()) → sampled action, log_prob sum, value model.py:97-100
         │    target weights: soft-max + cap30% redistribute excess loop env.py:70 for _ in range(n_assets): over=weights>cap+1e-7 excess sum → free weights free+=excess*free/sum
         │
         ↓  mean (49) = actor(state)  (deterministic inference) or sampled via _dist
         │  PPO update ppo.py:17-61 gamma0.99 gae_lambda0.95 clip0.2 value_coef0.5 entropy_coef0.01 max_grad0.5 update_epochs4 rollout256(prod512) minibatch64(prod128) model.py:80 bf16 cast, advantages normalized (mean)/std+1e-8 ppo.py:18
         │     compute_gae advantages zeros → reversed t nonterminal1-dones[t] delta rewards[t]+gamma*values[t+1]*nonterminal -values[t] advantage=delta+gamma*gae*nonterminal*advantage
         │     ppo_update: randperm → minibatch → obs slice → ratio exp(new_logp-old) surrogate min(ratio*adv, clamp(ratio,1-clip,1+clip)*adv) → loss -surrogate +0.5*mse(value) -0.01*entropy → scaler/GradScaler fp16 or autocast bf16 no scaler
         │
         ↓  action → PortfolioEnv.step → reward (net - penalties) → next obs
              episode length ≈1995 steps full panel ~1068 honest train-end → terminated index>=len(prices)-2 env.py:64
```

**Why this shape:** LLM prior for sequences not prices. Encoders (`market` mean-pooled over 30×49, `portfolio` 50, `risk` 3) translate numbers into 3 token embeddings the frozen LLM already knows how to contextualize. PPO hard part is exploration: `log_std ~ -0.5` + Gaussian + clip0.2 prevents churn, entropy0.01 prevents collapse. Reward shaping explains negative `mean_reward -0.01` even when `mean_net +0.09%` — `0.10*vol +0.20*drawdown` dominates.

### 3.3 Training Loop (train.py:174) — A5000 Optimized

```
CLI: --data data/nifty50_prices.csv --window30 --timesteps25000(default) --output artifacts/finance --backbone ModelConfig --cost-bps10 --max-weight0.30 --train-end2022-12-31 --device cuda --dtype bf16 --amp True --compile? --rollout-steps256(prod512) --minibatch64(prod128) --eval-interval5000 --checkpoint-interval5000 train.py:23
 ↓ TF32 knobs train.py:46 allow_tf32 True cudnn True set_float32_matmul_precision high
 ↓ env_cfg FinanceRewardConfig + PortfolioEnv(csv,window,initial100k,reward_config,start_date,end_date) + cfg ModelConfig + ppo_cfg PPOConfig override minibatch/rollout
 ↓ use_amp = amp and cuda, amp_dtype bf16/fp16/float32, scaler GradScaler if fp16 else None train.py:62
 ↓ policy InternLMFinancePolicy(cfg, n_assets49).to(device) + optional torch.compile train.py:68-78
 ↓ optimizer Adam 3e-4 train.py:80 + env.reset → obs
 ↓ while steps<timesteps batch collect rollout_steps with torch.no_grad autocast inference act tensor_obs train.py:90-100 (Triton fallback unwrap retry train.py:102) action bf16→float→cpu numpy env.step
 ↓ bootstrap policy(tensor_obs) → values stack advantages returns compute_gae ppo_update train.py:123-131
 ↓ metrics _metrics(infos) mean_net max_dd mean_turnover train.py:16 + print steps mean_reward mean_net max_dd turnover peak_equity + vram alloc/reserved train.py:132-139
 ↓ checkpoint if steps>=next_ckpt save_artifacts checkpoint_{steps} train.py:141-145 (7 dirs 10240→70144 interval ~10240)
 ↓ holdout eval if steps>=next_eval and train_end try evaluate_policy metrics_from_infos run_episode eval_env PortfolioEnv start2023→2024 same costs autocast deterministic infos metrics baseline equal-weight zero action → print [Holdout learned vs equal_weight alpha @steps] train.py:146-168
 ↓ env.close save_artifacts output serializable cfg ppo_cfg env_cfg tickers
```

**Hardware** `REPORT.md:120` RTX A5000 WDDM driver595.95 CUDA13.2 sm_86 24,564 MiB 25.8GB torch2.11+cu128 bf16 TF32 3.90GB alloc /6.05GB reserved for49 assets (1.8B≈3.6GB+LoRA/heads) Eager autocast smoke512 ~35-70s full25k ~27-45min prod75k ~1.3h ~70 episodes train-split.

### 3.4 File Map

| Block | File | Line |
|-------|------|------|
| Model/PPOConfig | `config.py` | `config.py:6-32` hidden2048 r16 wqkv/wo bf16 PPO γ0.99 λ0.95 clip0.2 |
| Legacy YAML | `config.yaml` | `config.yaml:4` stale [q_proj,v_proj] |
| Env (pivot ffill dropna, cap loop, reward) | `env.py` | `env.py:45` pivot, `70` cap, `63` reward |
| Encoders + InternLM LoRA + heads | `model.py` | `model.py:12` encoders, `36` remap, `85` hidden_states, `104` save |
| GAE + PPO | `ppo.py` | `ppo.py:6` compute_gae, `17` ppo_update |
| A5000 loop + holdout | `train.py` | `train.py:46` TF32, `90` rollout, `141` ckpt, `146` holdout |
| Holdout evaluator (nested adapter) | `evaluate_policy.py` | `evaluate_policy.py:36` nested rglob, `75` run_episode, `94` metrics |
| Baseline (equal 1/49) | `evaluate_baselines.py` | `evaluate_baselines.py:10` zeros |
| Data bundler | `prepare_nifty50_data.py` | `prepare_nifty50_data.py:15` NIFTY50 URL |
| Data | `data/nifty50_prices.csv` 287k rows + `nifty50_summary_statistics.csv` + `data/README.md` | `data/README.md:8` |

---

## 4. LOGICAL REASONING — 4TH DOMAIN (Not Started)

> Placeholder for same backbone. No code, no report.

```
Logical Reasoning (future) — reuses all three patterns:
  Dataset: reasoning traces (e.g., GSM8K-style <plan><step><verify><answer>) with keep-richest dedupe like deepfake dataset.py:179
  Model: same InternVL/InternLM backbone + new LoRA adapter_name="logic" r16/64 (r64 for reasoning depth) target ["wqkv","wo"] or ["q/k/v/o…"] matching backbone
  Training: weights_from current best adapters (e.g., weights_from finance prod) or fresh frozen, lr1e-5 lm_loss_weight0.3, trim_after_answer like inference.py:106
  Evaluation: holdout by reasoning type (not random), like TEP fault-disjoint / HydraFake per-image / Finance chronological
  Goal: 4 named adapters co-hosted on one frozen 1.8-2B core via Peft routing, usable as portfolio foundation for research/paper-trading + chemical forensics + media forensics + reasoning.
```

---

*End Architectures-Only — diagrams + explanatory text for all 3 (+1 future) with file:line provenance. See `F:/C68_CONTEXT_BOMB.md:80` for full context bomb (results, logs, JSONs, CSVs) and `F:/COMBINED_REPORT.md:1` for human-readable combined report.*
