# Recovery archives

Run `git lfs pull` after cloning. These ZIPs are real LFS objects, not ordinary Git blobs.

- research-and-demo-history-20261116.zip:66k+ historical non-weight files from the local research and demo workspaces, including source snapshots, reports, raw daily evaluations, training metadata and console logs. Snapshot predates the newly requested paper-trading demo; current tested implementation will be published separately.
- selected-checkpoint-packages-20261116.zip: selected package for every campaign with a recoverable selection record, plus the selected explanation adapter. Includes model weights, heads/configs and manifests. Source paths inside are relative to `research/` and `demo/`.
- snapshot_manifest.json and selected_checkpoint_manifest.json: SHA256 and original byte lengths for archived files. excluded_files.json records omitted weights/older ZIPs. Excluded non-selected intermediate weights remain local; this is NOT a complete backup of every28GB checkpoint or the multi-gigabyte base/virtual environments. Base is independently downloaded from the pinned upstream revision.

Extract to a new empty recovery directory, not over the active checkout. Install dependencies and use the new portable launcher when published; historical source paths in old receipts intentionally retain their original values for provenance. Current source paths and archived source paths differ and must not be confused.
