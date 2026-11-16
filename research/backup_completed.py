"""Copy completed campaign evidence to a new SSD folder; never touch originals."""
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).parent
TARGET=Path('F:/Polyvalent_Continuation_2026-11-07')
marker=TARGET/'CAMPAIGN_BACKUP.json'
if TARGET.exists() and not marker.exists():
    raise RuntimeError('Refusing an unrecognized pre-existing backup directory')
TARGET.mkdir(exist_ok=True)
if not marker.exists():
    marker.write_text(json.dumps({'source':str(ROOT),'created_utc':datetime.now(timezone.utc).isoformat()},indent=2))
for path in ROOT.iterdir():
    if path.is_file() and path.suffix in ('.py','.md','.txt','.json'):
        shutil.copy2(path,TARGET/path.name)
def tree(source,destination):
    shutil.copytree(source,destination,dirs_exist_ok=True,
        ignore=shutil.ignore_patterns('__pycache__','.venv','local_models'))
for name in ('covariance_trial','contracts','data','tests','checks','reports','deliverables'):
    if (ROOT/name).is_dir():tree(ROOT/name,TARGET/name)
completed=[]
for path in (ROOT/'campaign_runs').iterdir():
    if path.is_file():
        (TARGET/'campaign_runs').mkdir(exist_ok=True);shutil.copy2(path,TARGET/'campaign_runs'/path.name)
    elif any((path/file).exists() for file in ('selection.json','learning_check.json','summary.json','result.json')):
        tree(path,TARGET/'campaign_runs'/path.name);completed.append(path.name)
for path in (ROOT/'launch_records').iterdir():
    try:status=json.loads((path/'status.json').read_text())
    except (FileNotFoundError,json.JSONDecodeError):continue
    if status.get('status') in ('completed','failed'):tree(path,TARGET/'launch_records'/path.name)
(TARGET/'BACKUP_STATUS.json').write_text(json.dumps({'copied_utc':datetime.now(timezone.utc).isoformat(),
    'completed_run_folders':completed,'excluded':'virtual environment, large pristine base, in-progress run folders',
    'source_model_reference':'F:/pc/LY_C68/deepfake_project/models/InternVL3-2B'},indent=2))
print(json.dumps({'backup':str(TARGET),'completed_runs':len(completed)},indent=2))
