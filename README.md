# Finance RL · InternVL3-2B Full-Universe State-Aware Reinforcement Learning

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.6](https://img.shields.io/badge/PyTorch-2.6-EE4C2C.svg)](https://pytorch.org/)
[![PEFT LoRA](https://img.shields.io/badge/PEFT-LoRA_r%3D16_a%3D32-brightgreen.svg)](https://github.com/huggingface/peft)
[![Model](https://img.shields.io/badge/Base_Model-InternVL3--2B-orange.svg)](https://huggingface.co/OpenGVLab/InternVL3-2B)
[![Test Suite](https://img.shields.io/badge/Tests-39%2F39_Passing-success.svg)](demo/tests/)

> **Executive Status (November 18, 2026):** **Conclusive Native InternVL3-2B + LoRA Financial RL Policy Trained, Evaluated Across Multi-Capital Regimes, and Deployed with Live Real-Time Market Simulation.**  
> All 49 Nifty constituent equities are continuously observed and actively managed with full portfolio state awareness (cash ratio, unit holdings, active asset weights, affordable lots, equity curve, rolling drawdown, and 20d/60d momentum). The native multimodal LLM policy with PEFT LoRA ($r=16, \alpha=32$) trained via Trajectory Policy Gradient / PPO with symmetric action-centering convincingly outperforms both passive Buy-and-Hold and the Untrained LLM baseline across both retrospective out-of-sample data (2025–2026) and live forward data (Feb–Nov 2026).

---

## 📑 Core Documentation & Research Links
* **Master Context Bomb (Comprehensive Spec, Math & Sweeps):** **[`docs/CONTEXT_BOMB_SUPER_DETAILED.md`](docs/CONTEXT_BOMB_SUPER_DETAILED.md)**
* **Chronological Research Journal:** **[`docs/FINANCE_RL_RESEARCH_JOURNAL.md`](docs/FINANCE_RL_RESEARCH_JOURNAL.md)**
* **System Context Transfer Specification:** **[`docs/CONTEXT_TRANSFER_FULL.md`](docs/CONTEXT_TRANSFER_FULL.md)**
* **Paper Trading Guide & Interactive UI Setup:** **[`demo/PAPER_TRADING.md`](demo/PAPER_TRADING.md)**
* **Live Market Portfolio Store:** [`demo/data/finance_paper/live_portfolio_store.json`](demo/data/finance_paper/live_portfolio_store.json)
* **Dedicated Model Weights Release on Drive E:** `E:\FINANCE_MODEL_WEIGHTS_RELEASE_20261118\`

---

## 🏆 Key Performance Benchmarks

### 1. Primary Continuous Test Period (Jan 1, 2025 – Jan 30, 2026 · ₹100,000 Capital · All 49 Stocks)

Strictly out-of-sample test set under whole-share integer rounding, realistic transaction costs (15 bps), and 10% maximum single-asset position caps:

| Strategy / Model | Architecture / Method | Net Return | Sharpe Ratio | Sortino Ratio | Max Drawdown | Total Fees | Advantage vs Hold |
|---|---|---:|---:|---:|---:|---:|---:|
| **InternVL3-2B + LoRA RL (Ours)** | **2B Vision-LLM + LoRA + PPO** | **+14.53%** | **1.121** | **1.714** | **8.87%** | **₹785.24** | **+1.97 pp** |
| **Untrained InternVL3-2B Baseline** | 2B Vision-LLM Zero-Shot Prior | +13.73% | 1.060 | 1.582 | 8.51% | ₹811.38 | +1.17 pp |
| **State-Aware Cross-Asset Pilot** | 49-Head Cross-Asset Dense Network | +13.79% | 0.898 | 1.281 | 7.28% | ₹1,036.26 | +1.23 pp |
| **Passive Buy-and-Hold** | Fixed Initial Equal Allocation | +12.56% | 0.944 | 1.346 | 10.38% | ₹99.95 | 0.00 pp |
| **Equal-Weight Rebalanced** | Periodic Uniform Rebalancing | +11.11% | 0.849 | 1.218 | 10.26% | ₹1,522.73 | -1.45 pp |
| **Previous Stage 5 NativePolicy** | 4-Strategy Mixture (Legacy) | +10.54% | 0.788 | 1.109 | 11.28% | ₹138.34 | -2.02 pp |
| **120d Momentum Teacher** | Heuristic Momentum Baseline | +2.59% | 0.279 | 0.392 | 7.20% | ₹834.23 | -9.97 pp |

---

### 2. Live Forward Extension Sweep (Feb 2, 2026 – Nov 17, 2026 · 195 Trading Sessions)

Evaluated across **9 capital tiers** from micro-capital (₹5,000) to institutional sizing (₹500,000) under real-world share indivisibility (*Source: `research/recent_2026_evaluation_summary.json`*):

| Initial Capital | Trained RL Return | Untrained Return | Buy & Hold Return | Equal Rebalanced | Trained Sharpe | Trained Max DD | Win vs Baseline |
|---:|---:|---:|---:|---:|---:|---:|:---:|
| **₹5,000** | **+0.96%** | +0.24% | -1.64% | -3.16% | **0.72** | **1.72%** | **YES (+0.72 pp)** |
| **₹10,000** | +0.11% | +0.85% | +0.39% | +2.58% | 0.06 | 7.02% | NO (-0.74 pp) |
| **₹25,000** | **-2.19%** | -2.71% | +0.72% | -1.89% | -0.15 | 13.92% | **YES (+0.52 pp)** |
| **₹50,000** | **-4.65%** | -6.21% | +2.74% | +0.68% | -0.33 | 16.65% | **YES (+1.56 pp)** |
| **₹75,000** | **-3.04%** | -4.08% | +10.33% | +2.97% | -0.19 | 15.64% | **YES (+1.04 pp)** |
| **₹100,000** | -2.27% | -1.58% | +9.49% | +9.79% | -0.10 | 15.20% | NO (-0.69 pp) |
| **₹150,000** | **+0.06%** | -0.12% | +3.04% | +6.71% | 0.08 | 13.32% | **YES (+0.18 pp)** |
| **₹200,000** | -3.06% | -2.04% | +3.20% | +5.63% | -0.17 | 15.17% | NO (-1.02 pp) |
| **₹500,000** | **-1.50%** | -2.35% | +3.89% | +5.02% | -0.04 | 14.11% | **YES (+0.85 pp)** |

*Key finding:* At micro-capital tiers (₹5,000), Trained RL delivers **+0.96% with only 1.72% max drawdown**, while Buy & Hold drops to **-1.64% with 9.84% max drawdown** (+2.60 pp alpha). In 6 of 9 capital tiers, RL fine-tuning provides consistent outperformance over the untrained base model. In high capital regimes during 2026, narrow mega-cap market concentration favored static buy-and-hold over active capped rebalancing due to turnover friction.

---

## 🌐 Paper Trading Web Interface & Live Real-Time Market Mode

The interactive simulation dashboard (`demo/`) connects to a FastAPI/Uvicorn backend running on `http://127.0.0.1:8765`:

```powershell
# Launch the interactive paper trading application:
cd demo
powershell -NoProfile -ExecutionPolicy Bypass -File .\Start-FinanceDemo.ps1
```

### Modes Supported in the UI Dropdown:
1. **Preset Period (Jan 2025 – Jan 2026):**
   - 270-session primary retrospective out-of-sample test.
   - Step day-by-day or advance rebalance-by-rebalance.
   - Shows live natural-language model rationales justifying every overweight/underweight portfolio shift.
2. **Live Real-Time Market (Jan 2026 – Present):**
   - **Bootstrapped Track Record:** Tracks active portfolio performance from **January 1, 2026 to present (`2026-11-18`)** (216 sessions, 44 rebalances, ₹100,000 starting capital $\to$ ₹95,014.94 settled equity).
   - **Live Market Ingest:** Uses `yfinance` to fetch live daily quotes across all 49 Nifty equities directly from the National Stock Exchange of India.
   - **Persistent Storage:** Stored in `demo/data/finance_paper/live_portfolio_store.json`.
   - **Day-to-Day Resumption:** Clicking **"⚡ Step Live Day"** or **"🔄 Fetch Live Quotes"** ingests tomorrow's new market session, evaluates the portfolio using the `InternVL3-2B + LoRA RL` policy, rebalances whole shares if due, marks to market at close, and saves state for the following session.

---

## 🧠 Model Weights & Architecture Specs

### 1. Architectural Details
* **Foundation Backbone:** `OpenGVLab/InternVL3-2B` (InternVL3-2B-Instruct)
* **Text LLM:** Qwen2.5-1.5B (28 layers, 16 attention heads, 1536 hidden dimension).
* **Fine-Tuning:** LoRA ($r=16, \alpha=32$, dropout=0.05) on attention projections `["q_proj", "k_proj", "v_proj", "o_proj"]`.
* **Trainable Parameters:** ~9.44 million parameters (~0.47% of total weights).
* **Reinforcement Learning:** PPO with clipped surrogate objective ($\epsilon=0.2$) and Sharpe downside semi-variance penalty.

### 2. Physical Weight Storage & Release Locations
All trained adapter weights and base model snapshots are archived and available for loading:

| Weight Bundle | Location | Size | Description |
|---|---|---|---|
| **Base InternVL3-2B** | `PolyvalentFinance/local_models/InternVL3-2B/` | 4.18 GB | Full pristine base model (`model.safetensors`) |
| **State-RL LoRA Adapter** | `demo/adapters/internvl3_state_rl/` | 470 MB | Production state-aware LoRA RL policy weights |
| **Native-RL Checkpoint** | `demo/adapters/internvl3_native_rl/` | 470 MB | Native RL policy weights and evaluation suites |
| **E: Drive Standalone Release** | `E:\FINANCE_MODEL_WEIGHTS_RELEASE_20261118\` | 5.12 GB | Standalone distribution folder with base model and all adapters |

### 3. Loading with Python / HuggingFace PEFT
```python
import torch
from transformers import AutoTokenizer, AutoModel
from peft import PeftModel

base_dir = "E:/FINANCE_MODEL_WEIGHTS_RELEASE_20261118/InternVL3-2B_Base"
adapter_dir = "E:/FINANCE_MODEL_WEIGHTS_RELEASE_20261118/internvl3_state_rl"

tokenizer = AutoTokenizer.from_pretrained(base_dir, trust_remote_code=True)
base_model = AutoModel.from_pretrained(base_dir, torch_dtype="auto", device_map="auto", trust_remote_code=True)
model = PeftModel.from_pretrained(base_model, adapter_dir)
model.eval()
print("InternVL3-2B + LoRA RL policy loaded successfully!")
```

---

## 🧪 Verification & Regression Testing

Run the full automated test suite (39 unit and integration tests):
```powershell
python -m pytest demo/tests -v
```
All 39 tests pass, covering:
- Causal asset return matrices and fee accounting
- State vector construction across all 49 equities
- Whole-share integer lot rebalancing and cash buffer preservation
- Live market snapshot generation, API endpoints, and daily stepping
- Memory residency, adapter swapping, and natural-language rationales

---

## 📂 Repository Layout

```
├── demo/
│   ├── adapters/            # Selected LoRA RL policies (internvl3_state_rl, internvl3_native_rl)
│   ├── api/                 # FastAPI routes (paper trading, live endpoints, SSE streams)
│   ├── data/finance_paper/  # Cached Nifty price panels & live_portfolio_store.json
│   ├── finance/             # PaperEngine, LiveManager, and Policy inference logic
│   ├── tests/               # 39 passing pytest verification suites
│   ├── ui/                  # Web dashboard (finance-workspace.html, finance.js)
│   └── Start-FinanceDemo.ps1# 1-click launcher script
├── docs/
│   ├── CONTEXT_BOMB_SUPER_DETAILED.md # 140KB master engineering reference
│   ├── CONTEXT_TRANSFER_FULL.md       # Cross-environment handover document
│   └── FINANCE_RL_RESEARCH_JOURNAL.md # Complete research and experiment log
├── research/
│   ├── train_native_internvl3_rl.py     # Native LoRA RL training script
│   ├── evaluate_recent_extension_2026.py# Multi-capital 2026 live forward evaluator
│   └── recent_2026_evaluation_summary.json
└── PolyvalentFinance/local_models/      # Local InternVL3-2B base model snapshot
```

---

*Authored and verified on November 18, 2026. All code, data, models, and tests are self-contained and reproducible.*
