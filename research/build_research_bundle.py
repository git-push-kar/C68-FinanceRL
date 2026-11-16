"""Package frozen models, runtime and auditable evidence after final evaluation."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
from zipfile import ZipFile,ZIP_DEFLATED
from execution_policy import wait_for_compute_window

ROOT=Path(__file__).parent.resolve()


def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--results',required=True)
    p.add_argument('--output',required=True);args=p.parse_args()
    results=Path(args.results).resolve();protocol=json.loads((results/'frozen_protocol.json').read_text())
    if json.loads((results/'status.json').read_text())['status']!='completed':raise ValueError('Final evidence not complete')
    output=Path(args.output).resolve();archive=output.with_suffix('.zip')
    if output.exists() or archive.exists():raise FileExistsError('Choose a new bundle path')
    output.mkdir(parents=True)
    def copy(path):
        path=Path(path).resolve();relative=path.relative_to(ROOT);target=output/relative
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
    # Frozen runtime and tests; preserve campaign-relative paths for auditability.
    for name,expected in protocol['runtime_source_sha256'].items():
        path=ROOT/'covariance_trial'/name
        if sha(path)!=expected:raise ValueError('Runtime changed since final freeze: '+name)
        copy(path)
    for name,expected in protocol['runtime_contract_sha256'].items():
        if sha(ROOT/'covariance_trial'/name)!=expected:raise ValueError('Base contract changed since final freeze')
    for name in ('contracts','tests'):
        for path in (ROOT/'covariance_trial'/name).rglob('*'):
            if path.is_file() and '__pycache__' not in path.parts:copy(path)
    for path in (ROOT/'covariance_trial').glob('*.md'):copy(path)
    for path in ROOT.iterdir():
        if path.is_file() and path.suffix in ('.py','.md','.txt','.json'):copy(path)
    for folder in ('reports','checks','contracts'):
        for path in (ROOT/folder).rglob('*'):
            if path.is_file() and '__pycache__' not in path.parts:copy(path)
    copy(ROOT/protocol['data'])
    for path in results.glob('*.json'):copy(path)
    # Keep every attempt's recorded outcomes and exact source, including failures.
    for folder in (ROOT/'campaign_runs').iterdir():
        if folder.is_file():copy(folder);continue
        for path in folder.glob('*.json'):copy(path)
        if (folder/'source_snapshot').exists():
            for path in (folder/'source_snapshot').rglob('*'):
                if path.is_file():copy(path)
    for folder in (ROOT/'launch_records').iterdir():
        state=json.loads((folder/'status.json').read_text())
        if state['status'] not in ('completed','failed'):raise ValueError('Campaign job is still active: '+folder.name)
        for path in folder.rglob('*'):
            if path.is_file() and '__pycache__' not in path.parts:copy(path)
    # Include all sealed selected artifacts and matched controls, not just a winner.
    for entry in protocol['entries']:
        if entry['kind']=='internvl':
            package=ROOT/entry['package'];manifest=json.loads((package/'manifest.json').read_text());copy(package/'manifest.json')
            for name,expected in manifest['sha256'].items():
                if sha(package/name)!=expected:raise ValueError('Artifact changed: '+str(package/name))
                copy(package/name)
        else:
            path=ROOT/entry['weights']
            if sha(path)!=entry['sha256']:raise ValueError('Control changed: '+str(path))
            copy(path)
    deployment_seed=protocol['deployment_seed']
    primary=next(e['package'] for e in protocol['entries'] if e['name']==f'candidate_seed{deployment_seed}')
    text=f'''# Portable finance specialist and research evidence

Selected family: {protocol['selected_family']}. Representative package: `{primary}`.
All three frozen seeds and controls are included. The default seed {deployment_seed}
was selected by lowest pre-2023 risk loss, before final-period outcomes.
Seed 42 remains a fixed technical/timing example. Read reports/FINAL_REPORT.md
and reports/METHODS_AND_LIMITS.md before interpreting the numbers.

The shared pristine 4.18-GB base and virtual environment are excluded. The base
ID/revision/file hashes are in contracts/base_snapshot.json. On this PC its
verified source remains F:/pc/LY_C68/deepfake_project/models/InternVL3-2B.
Do not substitute a reasoning-merged base or a standalone Qwen model.

Create a new Python 3.12 environment. Install torch 2.6.0+cu124 and torchvision
0.21.0+cu124 from the official PyTorch cu124 index, then install the remaining
resolved packages in ENVIRONMENT_LOCK_20261107.txt. CUDA compatibility and other
hardware require validation; copying the Windows venv is not supported.

Set FINANCERL_BASE_MODEL to the verified pristine snapshot, set
FINANCERL_EXECUTION=selective_fp32, HF_HUB_OFFLINE=1, TRANSFORMERS_OFFLINE=1,
OMP_NUM_THREADS=2 and MKL_NUM_THREADS=2. Runtime scripts are in covariance_trial.
They pause heavy compute from 22:50 to 07:00 IST. The payload schema includes
the exact ticker order, 30x49x5 market features, 49x49 causal covariance,
holdings/cash, risk state and trading_step for the 21-session cadence.
For the reported final protocol, set signal_lag_closes=1, provide the entire
previous-close observation and its signal_close_date, the current
execution_close_date, and separate actual current execution_portfolio weights.
The caller supplies the exchange calendar and correctly delayed features.
The final historical window is May 2025 through 30 January 2026, starting cash;
the earlier full-2025 plan was changed before final outcome access after
verifying the pinned base weight upload date of 10 April 2025.

From covariance_trial, run inference.py with --package pointing to the complete
package, --request pointing to the copied checks/router_smoke_deployment/request.json,
and --output pointing to a fresh JSON file. Integration example and exact
development commands remain in launch_records. The finance_lora file alone
is insufficient: finance_heads.pt, model/environment configuration and schema
are required. The specialist outputs allocations, not financial QA text.

MANIFEST_SHA256.json covers every included file except itself. Archived
post-publication outcomes are a used final block; do not retune and describe it as unseen.
The root scripts preserve historical code; do not run old automatic benchmark
drivers. Exact attempt source/logs and all selected final-control weights
are retained. Unselected large checkpoint weights remain in the full SSD
campaign backup rather than this portable archive.
'''
    (output/'BUNDLE_README.md').write_text(text,encoding='utf-8')
    files={p.relative_to(output).as_posix():sha(p) for p in output.rglob('*') if p.is_file()}
    (output/'MANIFEST_SHA256.json').write_text(json.dumps(files,indent=2))
    wait_for_compute_window()
    with ZipFile(archive,'x',compression=ZIP_DEFLATED,compresslevel=6) as z:
        for path in sorted(output.rglob('*')):
            if path.is_file():z.write(path,path.relative_to(output).as_posix())
    with ZipFile(archive) as z:
        if z.testzip() is not None:raise RuntimeError('Archive CRC verification failed')
    print(json.dumps({'directory':str(output),'archive':str(archive),'bytes':archive.stat().st_size,
        'sha256':sha(archive),'files':len(files)+1,'base_weights_included':False},indent=2))


if __name__=='__main__':main()
