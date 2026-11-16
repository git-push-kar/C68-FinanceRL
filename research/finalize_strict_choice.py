"""Record deterministic pre-2023 checkpoint selection after strict replicas finish."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).parent
if (ROOT/'reserved_finance_2025_test.lock.json').exists():raise RuntimeError('Cannot change model registry after final testing')
path=ROOT/'METHOD_CHOICE.json';choice=json.loads(path.read_text())
if choice['selected_family']!='covariance_risk_strict_seed':raise ValueError('Unexpected method family')
steps={};artifacts={};primary_risk={}
for family in ('covariance_risk_strict_seed','covariance_risk_strict_no_lora_seed'):
    for seed in (42,123,456):
        folder=ROOT/'campaign_runs'/f'{family}{seed}'
        if json.loads((folder/'status.json').read_text())['status']!='completed':raise ValueError('Training incomplete')
        selection=json.loads((folder/'selection.json').read_text())
        candidates=json.loads((folder/'candidates.json').read_text())
        expected=min(candidates,key=lambda r:(r['fidelity']['mean_log_relative_variance'],r['step']))
        if selection['selected']['step']!=expected['step']:raise ValueError('Selection did not follow the fixed risk rule')
        package=(ROOT/'covariance_trial'/selection['selected']['package']).resolve()
        manifest=json.loads((package/'manifest.json').read_text())
        numerical=manifest['run']['numerical_backend']
        if numerical!={'float32_matmul_precision':'highest','cuda_matmul_allow_tf32':False,'cudnn_allow_tf32':False}:
            raise ValueError('Replica did not use the required strict numerical backend')
        artifacts[f'{family}{seed}']={'step':expected['step'],
            'package':package.relative_to(ROOT).as_posix(),
            'manifest_sha256':hashlib.sha256((package/'manifest.json').read_bytes()).hexdigest()}
        if family==choice['selected_family']:
            steps[str(seed)]=expected['step']
            primary_risk[str(seed)]=expected['fidelity']['mean_log_relative_variance']
deployment_seed=min((42,123,456),key=lambda seed:(primary_risk[str(seed)],seed))
choice.update(selected_steps=steps,strict_artifacts=artifacts,
    deployment_seed=deployment_seed,deployment_selection_risk=primary_risk,
    deployment_selection_rule='Lowest pre-2023 mean log relative variance across the three selected checkpoints; lowest seed on tie. Primary paper comparison still reports all three. Representative seed42 remains a fixed technical example. No final-period return selection.',
    checkpoint_registry_finalized_utc=datetime.now(timezone.utc).isoformat(),
    registry_basis='Only fixed pre-2023 risk selection and strict-backend validation; no financial outcome selection.')
path.write_text(json.dumps(choice,indent=2));print(json.dumps({'selected_steps':steps,'strict_packages':len(artifacts),'pre23_deployment_seed':deployment_seed},indent=2))
