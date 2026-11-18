# Historical checkpoint recovery

The current runnable demo is `demo/`, not an extracted historical experiment.
Install Git LFS, clone `main`, then run `git lfs pull`. The pinned pretrained
InternVL3-2B base is separately downloaded as described in `demo/PAPER_TRADING.md`.
Python environments, credentials, upstream base weights and caches are excluded.

Two LFS ZIPs preserve historical metadata/source and selected checkpoint packages.
For a complete historical working tree, extract the metadata ZIP into a **new empty
directory** and then restore weights into that same directory using the release
archive. Never extract historical files over the active repository checkout.

The checkpoint release contains all 5,224 weight-file paths found under the local
research/demo trees, excluding upstream base/environment/cache directories.
It stores 4,663 unique byte contents once, mapped back to their original paths in
`weights-index.json`. This includes unsuccessful and intermediate checkpoints;
their inclusion does not make them selected or recommended models.

Release: https://github.com/git-push-kar/C68-FinanceRL/releases/tag/finance-checkpoint-archive-20261116

Download `asset-manifest.json` into a new asset-cache folder. Compare its SHA256
against the copy committed beside this document, then run with Python 3.11+:

```powershell
python archives/restore_checkpoint_weights.py --assets C:/FinanceAssetCache --destination C:/FinanceRecovered
```

The helper downloads only required ZIP parts, verifies their full SHA256 and size,
verifies each restored object's SHA256 and size, and refuses to replace a differing
existing file. Allow substantial disk space for both compressed assets and restored
files (30.3 GB of original weights plus source/metadata). For selective recovery:

```powershell
python archives/restore_checkpoint_weights.py --assets C:/FinanceAssetCache --destination C:/FinanceRecovered --only "research/campaign_runs/asset_allocation_dynamic_seed123_64/checkpoint_0/numeric.pt"
```

`--offline` requires all matching assets already downloaded. The helper does not
load checkpoint code or unpickle tensors. Historical `.pt` files should only be
loaded after reviewing their provenance. GitHub release receipts and the committed
asset manifest provide integrity evidence; they do not independently validate a
model's financial performance.
