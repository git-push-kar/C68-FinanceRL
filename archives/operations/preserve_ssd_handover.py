"""Machine-specific preservation of the original SSD reports and handover."""
from pathlib import Path
import hashlib,json,re,zipfile
G=Path('C:/Users/Parth/Documents/C68-FinanceRL');S=Path('F:/')
O=G/'docs/ssd_historical_context';O.mkdir(exist_ok=True)
patterns=[rb'gh[pousr]_[A-Za-z0-9_]{25,}',rb'github_pat_[A-Za-z0-9_]{25,}',rb'hf_[A-Za-z0-9]{25,}',rb'sk-[A-Za-z0-9_-]{25,}',rb'AKIA[A-Z0-9]{16}']
manifest=[]
def check(p):
 b=p.read_bytes()
 if p.suffix.lower() in ['.md','.txt','.json','.yaml','.yml','.py','.ps1','.ini','.toml','.csv']:
  if any(re.search(pattern,b) for pattern in patterns):raise ValueError('Potential credential requires review: '+str(p))
 return b
for p in sorted(S.glob('*.md')):
 b=check(p);(O/p.name).write_bytes(b);manifest.append({'source':str(p),'destination':'docs/ssd_historical_context/'+p.name,'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()})
root=S/'Polyvalent_Handover_2026-11-07';target=G/'archives/original-ssd-handover-20261107.zip';members=[]
with zipfile.ZipFile(target,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as z:
 for p in sorted(root.rglob('*')):
  if not p.is_file() or any(x in ['.git','__pycache__','.venv'] for x in p.relative_to(root).parts) or p.name=='.env':continue
  b=check(p);name=p.relative_to(root).as_posix();z.writestr(name,b);members.append({'path':name,'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()})
with zipfile.ZipFile(target) as z:
 for row in members:
  b=z.read(row['path'])
  assert len(b)==row['bytes'] and hashlib.sha256(b).hexdigest()==row['sha256']
(G/'archives/ssd_handover_manifest.json').write_text(json.dumps({'root_reports':manifest,'archive':target.name,'archive_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'archive_bytes':target.stat().st_size,'members':members,'credential_check':'High-confidence token patterns found zero hits in selected plaintext files; not an exhaustive security assessment. Nested historical ZIPs are preserved as original bytes.','excluded':'Git object stores, environments, bytecode and .env files'},indent=2))
print('Preserved',len(manifest),'root reports and',len(members),'handover files; ZIP bytes',target.stat().st_size)
