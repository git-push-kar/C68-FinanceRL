# Checkpoint backup complete — 16 November 2026

**Published and publicly verified:**
https://github.com/git-push-kar/C68-FinanceRL/releases/tag/finance-checkpoint-archive-20261116

- 5,224 original checkpoint-file paths; 30,279,357,187 original bytes.
- 4,663 unique weight objects, all independently streamed and SHA256/size checked.
- 17 checkpoint ZIP parts; each is below GitHub's per-asset limit.
- 22 published assets including the index, manifest, restore helper and recovery
  instructions; total compressed asset bytes: 17,227,149,909.
- Every data asset's GitHub-reported SHA256 and size matched the local manifest.
- Anonymous public downloads of the manifest, index and restore helper matched
  the local originals. The public release API confirmed all 22 assets.
- A historical checkpoint was restored locally with independent file hash/size
  verification. The selected runnable adapter and data were separately recovered
  through Git LFS from a fresh GitHub clone and used for real model-backed playback.

See `published_checkpoint_receipt.json`, `asset-manifest.json`,
`object_verification.json`, [recovery instructions](CHECKPOINT_RECOVERY.md) and
[fresh-clone verification](../docs/RECOVERY_VERIFICATION.md).

The two current research/demo workspaces are covered by this complete weight
release plus their metadata/source archives. The original SSD Finance trees and
college handover have separate LFS archives and manifests. This is not a backup
of unrelated SSD domains, upstream base weights, environments, caches or secrets.
Pinned base recovery and dependency installation are documented in the demo.
