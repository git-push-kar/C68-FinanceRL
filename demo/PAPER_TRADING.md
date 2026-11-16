# Nifty paper-trading demonstration

The selected InternVL3-2B Nifty RL adapter (seed42, checkpoint44) controls the historical paper account. This replaces the older ETF replay as the Finance workspace. **Finance conversation is removed after failing factual-accuracy testing.** The decision inspector is a ledger/allocation calculation, not generated rationale.

## On this PC

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\Start-FinanceDemo.ps1
```

Open http://127.0.0.1:8765/#finance. Start simulation for autonomous playback, or use Next day / Next rebalance. Select View decision for exact transactions and strategy mix. Holdings/activity export the revealed records. Reset portfolio restarts from cash with the same history. Sessions are in memory; restart the server to discard them.

The full test spans1January2025–30January2026. No broker connection, today's prices or real money are involved. On every fifth session the actual selected LoRA performs inference. Its fixed preferences blend four causal strategy tools; changing lagged data and drift change stock targets and trades. If actual inference fails or differs from the selected artifact, the simulation stops rather than substituting recorded action probabilities.

## Recover on another PC

From the GitHub repository root:

```powershell
git lfs install
git lfs pull --include="demo/adapters/finance_policy/**,demo/data/finance_paper/**"
cd demo
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
.\.venv\Scripts\python.exe -m pip install -r requirements-paper-demo.txt
```

The tested environment is Windows/Python3.12, RTX4060Ti8GB. The launcher never installs dependencies or silently downloads a different model. Install the **full pinned original base** separately:

```powershell
.\.venv\Scripts\python.exe -c "from huggingface_hub import snapshot_download; snapshot_download('OpenGVLab/InternVL3-2B', revision='899155015275a9b7338c7f4677e19c784e0e5a21', local_dir='../models/InternVL3-2B')"
powershell -NoProfile -ExecutionPolicy Bypass -File .\Start-FinanceDemo.ps1 -BaseModelPath ..\models\InternVL3-2B
```

Download requires internet once. Inference then uses offline flags. The base's `model.safetensors` SHA256 is `18b4cdbc8eab790f447feac717f4d0cecbb85caf8df198b3b2401c22506858e7`. Include all snapshot configuration, tokenizer and remote-code files, not only its weight tensor. The distribution does not include a virtual environment or upstream base. Other-domain trained weights are not all present; this does not certify the four-domain system on a new PC.

On another OS, install corresponding PyTorch/CUDA packages, set `FINAL_BASE_DIR` to the pinned snapshot and run `python -m uvicorn api.server:app --host 127.0.0.1 --port 8765` from `demo`. The default policy path is repository-relative. CUDA/layout on this PC is the verified configuration; no CPU-speed or arbitrary hardware-parity guarantee is claimed.

## Verification

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q
```

The35-test suite checks account/fee parity, causal target construction, references, failure behavior, stale advances and residency/transport. The golden fixture comes from the original research outputs, not regenerated expected values. CPU reference tests explicitly inject recorded probabilities; they do not certify actual model inference. Actual model receipts are in `finance/evaluation/paper_actual_run_receipt.json` and the production-browser record in `finance/evaluation/paper_ui/actual_browser_verification.json`.

For a full fresh real-model API run, use `verify_paper_live.py` against the running local server. For UI checks, install Playwright/Chromium separately and use `verify_actual_paper_browser.py`. That browser test uses actual production calls; `test_paper_ui.py` is explicitly a transport-fixture test. Do not present fixture screenshots as real live inference. Do not run a second GPU model process concurrently with this server.

## Research interpretation

Training2019–2021; validation2022; retrospective continuous test2025–January2026. Raw return11.22%, trained13.28%, equal holding13.06%. Rebalanced equal weight13.44% is stronger. Holding superiority is uncertain and cost-sensitive. No development score is used as the headline. No future prices enter a trade target; account charts expose only advanced history. Vendor-adjusted prices, universe/preprocessing, retrospective selection and simplified fills remain limitations.

Full method/failure/context documentation is in the repository's `docs/JOURNAL_RESEARCH_REPORT.md` and `docs/CONTEXT_TRANSFER_FULL.md`. The latter preserves prior experiments, including negative results and obsolete statuses. Later dated decisions supersede historical final-candidate claims.
