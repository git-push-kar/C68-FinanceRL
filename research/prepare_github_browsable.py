from pathlib import Path
import shutil,json,hashlib
R=Path('C:/Users/Parth/Documents/PolyvalentFinance');G=R.parent/'C68-FinanceRL';docs=G/'docs';docs.mkdir(exist_ok=True)
V=R/'deliverables/finance_context_transfer_20261116_v5'
shutil.copy2(V/'CONTEXT_TRANSFER.md',docs/'CONTEXT_TRANSFER_FULL.md')
shutil.copytree(R/'reports/nifty_final_candidate_test_20261116',docs/'nifty_test_results',dirs_exist_ok=True)
shutil.copytree(R/'reports/profit_improving_runs_audit_20261116',docs/'run_audits',dirs_exist_ok=True)
shutil.copy2(R/'Finance_Final_Conclusive_Run_Results.md',docs/'FINAL_RESULTS_SHORT.md')
for name in ['FINANCE_FINAL_AUDIT_V3.md','FINANCE_MODEL_GENERATION.md']:
 shutil.copy2(R.parent/'PolyvalentDemo'/name,docs/name)
source=G/'research';source.mkdir(exist_ok=True)
for p in R.iterdir():
 if p.is_file() and p.suffix.lower() in ['.py','.yaml','.yml'] and p.name not in ['prepare_github_archive.py'] :shutil.copy2(p,source/p.name)
for name in ['rl_candidate_static_native','rl_risk_selector_native','rl_multiasset_native']:
 shutil.copytree(R/name,source/name,ignore=shutil.ignore_patterns('__pycache__'),dirs_exist_ok=True)
(G/'archives/README.md').write_text('''# Recovery archives

Run `git lfs pull` after cloning. These ZIPs are real LFS objects, not ordinary Git blobs.

- research-and-demo-history-20261116.zip:66k+ historical non-weight files from the local research and demo workspaces, including source snapshots, reports, raw daily evaluations, training metadata and console logs. Snapshot predates the newly requested paper-trading demo; current tested implementation will be published separately.
- selected-checkpoint-packages-20261116.zip: selected package for every campaign with a recoverable selection record, plus the selected explanation adapter. Includes model weights, heads/configs and manifests. Source paths inside are relative to `research/` and `demo/`.
- snapshot_manifest.json and selected_checkpoint_manifest.json: SHA256 and original byte lengths for archived files. excluded_files.json records omitted weights/older ZIPs. Excluded non-selected intermediate weights remain local; this is NOT a complete backup of every28GB checkpoint or the multi-gigabyte base/virtual environments. Base is independently downloaded from the pinned upstream revision.

Extract to a new empty recovery directory, not over the active checkout. Install dependencies and use the new portable launcher when published; historical source paths in old receipts intentionally retain their original values for provenance. Current source paths and archived source paths differ and must not be confused.
''',encoding='utf-8')
(docs/'START_HERE.md').write_text('''# Finance research: current status

Current selected research artifact: original InternVL3-2B Nifty native static strategy-preference LoRA, seed42 checkpoint44. Three-seed results and full trade/accounting evidence are in nifty_test_results/FINAL_TEST_REPORT.md. Training2019–2021; validation2022; retrospective tests2025–January2026. No development scores should be passed off as untouched test evidence.

Headline continuous test: raw11.221%, trained13.283%, equal-weight buy-and-hold13.059%. A rebalanced equal-weight comparator returned13.441%, so no classical superiority claim is established. Tiny advantage over holding is uncertain/cost-sensitive. Current running menu still shows the older ETF experiment. A real paper-trading menu for the selected Nifty adapter and a keep/remove explanation evaluation are in progress and are not yet certified.

Read FINAL_RESULTS_SHORT.md, nifty_test_results/FINAL_TEST_REPORT.md, then CONTEXT_TRANSFER_FULL.md for the full experiment history. Historical files sometimes call earlier candidates final; the current status and later dated audit supersede them. No report proves that results cannot improve or that the process is flawless.

Recovery: see ../archives/README.md. Credentials, virtual environments and upstream base weights are excluded. Do not claim all intermediate weights are backed up.
''',encoding='utf-8')
p=G/'README.md';old=p.read_text(encoding='utf-8');p.write_text('# Finance RL research and paper-trading demo\n\nBranch `main` contains the current finance research transfer. Start with [docs/START_HERE.md](docs/START_HERE.md), [the full handover](docs/CONTEXT_TRANSFER_FULL.md), and [recovery archives](archives/README.md).\n\nThe historical implementation below is retained for provenance; its model and status descriptions do not describe the selected native InternVL3-2B Nifty result. Paper-trading integration is currently in progress.\n\n---\n\n'+old,encoding='utf-8')
print('Prepared browsable reports, source and recovery instructions')
