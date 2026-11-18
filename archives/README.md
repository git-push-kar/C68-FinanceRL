# Recovery archives

Install Git LFS and run `git lfs pull` after cloning branch `main`. The ZIPs below are real LFS objects, not ordinary Git blobs.

- `research-and-demo-history-20261116.zip`: 66,432 historical non-weight files from the research and demo workspaces: source, reports, raw daily evaluations, training metadata and console logs. This snapshot predates the new paper-trading implementation; current tested code is in `../demo/`.
- `selected-checkpoint-packages-20261116.zip`: selected packages from recoverable campaign selection records, plus the selected explanation adapter. It preserves unsuccessful research history as well as selected candidates; inclusion is not endorsement.
- `original-ssd-handover-20261107.zip`: 3,004 original college/SSD handover files, including source snapshots, migration packages and reference PDF/DOCX/PPTX documents. Original root context bombs are also directly readable under `../docs/ssd_historical_context/`.
- `original-ssd-finance-trees-20261116.zip`: 192 original Finance source/data/artifact files from both SSD Finance trees, including the earlier college checkpoints. Its SHA256 and complete file inventory are in `legacy_finance_manifest.json`. Git metadata, Python environments and caches are excluded.
- `snapshot_manifest.json`, `selected_checkpoint_manifest.json` and `ssd_handover_manifest.json`: SHA256 and original byte lengths. `metadata_zip_verification.json` records the complete historical metadata ZIP check. `excluded_files.json` describes exclusions from the initial metadata-only snapshot; it is not the final all-checkpoint recovery inventory.

All intermediate checkpoint weights are handled separately by the [checkpoint release recovery guide](CHECKPOINT_RECOVERY.md), its `asset-manifest.json`, `weights-index.json` and published release receipt. The selected runnable policy also lives directly in `../demo/adapters/finance_policy/` via LFS, so using the demo does not require restoring every experimental checkpoint.

Extract historical ZIPs to a new empty recovery directory. Do not extract them over the active checkout: old code, paths and status claims are preserved for provenance. Use the current [portable demo setup](../demo/PAPER_TRADING.md) for execution. Upstream pristine base weights, Python environments, caches and credentials are excluded; the pinned public model download and base hash are documented separately. This repository backs up Finance work and its project handover references, not every unrelated file on the SSD.
