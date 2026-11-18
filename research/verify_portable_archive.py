"""Extract the actual archive, verify every hash, and run its default specialist."""
import argparse
import hashlib
import json
import os
from pathlib import Path,PurePosixPath
import subprocess
import sys
from zipfile import ZipFile

ROOT=Path(__file__).parent.resolve()


def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--archive',required=True)
    p.add_argument('--extract-to',required=True);p.add_argument('--output',required=True);args=p.parse_args()
    archive=Path(args.archive).resolve();destination=Path(args.extract_to).resolve();output=Path(args.output).resolve()
    if not destination.is_relative_to(ROOT) or destination==ROOT:raise ValueError('Extraction must be in a new campaign subdirectory')
    if destination.exists() or output.exists():raise FileExistsError('Use fresh extraction/result paths')
    with ZipFile(archive) as z:
        for item in z.infolist():
            path=PurePosixPath(item.filename)
            if path.is_absolute() or '..' in path.parts or any(':' in part for part in path.parts):
                raise ValueError('Archive path is not relative and safe')
        destination.mkdir();z.extractall(destination)
    manifest=json.loads((destination/'MANIFEST_SHA256.json').read_text())
    for name,expected in manifest.items():
        if sha(destination/name)!=expected:raise ValueError('Extracted file hash differs: '+name)
    choice=json.loads((destination/'METHOD_CHOICE.json').read_text());seed=choice['deployment_seed']
    package=destination/choice['strict_artifacts'][f"{choice['selected_family']}{seed}"]['package']
    request=destination/'checks/router_smoke_deployment/request.json'
    expected=json.loads((destination/'checks/router_smoke_deployment/prediction.json').read_text())
    prediction=ROOT/'reports/portable_prediction.json';log=ROOT/'reports/portable_inference.log'
    if prediction.exists():raise FileExistsError(prediction)
    env=dict(os.environ);env.update(FINANCERL_EXECUTION='selective_fp32',
        FINANCERL_BASE_MODEL=str(ROOT/'local_models/InternVL3-2B'),HF_HUB_OFFLINE='1',
        TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
    command=[sys.executable,str(destination/'covariance_trial/inference.py'),'--package',str(package),
        '--request',str(request),'--output',str(prediction),'--device','cuda']
    with log.open('w',encoding='utf-8') as stream:
        subprocess.run(command,cwd=destination/'covariance_trial',env=env,stdout=stream,stderr=subprocess.STDOUT,
            check=True,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
    actual=json.loads(prediction.read_text());error=max(abs(a-b) for a,b in zip(expected['weights'],actual['weights']))
    if expected['tickers']!=actual['tickers'] or expected['base_revision']!=actual['base_revision'] or error>1e-7:
        raise AssertionError('Portable prediction does not reproduce the verified deployment example')
    result={'status':'passed','archive':str(archive),'archive_sha256':sha(archive),'extracted':str(destination),
        'files_verified':len(manifest),'deployment_seed':seed,'maximum_weight_difference':error,
        'same_base_revision':actual['base_revision'],'scope':'Actual extracted runtime and selected weights; original pinned base supplied externally; development example, no new final-period evaluation.'}
    output.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':main()
