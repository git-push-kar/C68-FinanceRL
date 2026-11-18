"""Preserve the two original Finance SSD trees, apart from Git/env/cache data."""
from pathlib import Path
import hashlib,json,re,zipfile
G=Path('C:/Users/Parth/Documents/C68-FinanceRL');rows=[]
target=G/'archives/original-ssd-finance-trees-20261116.zip'
patterns=[rb'gh[pousr]_[A-Za-z0-9_]{25,}',rb'github_pat_[A-Za-z0-9_]{25,}',rb'hf_[A-Za-z0-9]{25,}',rb'sk-[A-Za-z0-9_-]{25,}',rb'AKIA[A-Z0-9]{16}']
excluded=[]
with zipfile.ZipFile(target,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as z:
 for label,root in [('FINANCE/C68-FinanceRL',Path('F:/FINANCE/C68-FinanceRL')),('C68-FinanceRL',Path('F:/C68-FinanceRL'))]:
  for p in sorted(root.rglob('*')):
   if not p.is_file():continue
   rel=p.relative_to(root)
   if any(x in ['.git','.venv','venv','__pycache__','.pytest_cache','.cache','local_models'] for x in rel.parts) or p.name.startswith('.env'):
    excluded.append(label+'/'+rel.as_posix());continue
   b=p.read_bytes()
   if p.suffix.lower() in ['.md','.txt','.json','.yaml','.yml','.py','.ps1','.ini','.toml','.csv'] and any(re.search(x,b) for x in patterns):raise ValueError('Potential credential requires review: '+str(p))
   name=label+'/'+rel.as_posix();z.writestr(name,b);rows.append({'path':name,'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()})
with zipfile.ZipFile(target) as z:
 for r in rows:
  b=z.read(r['path']);assert len(b)==r['bytes'] and hashlib.sha256(b).hexdigest()==r['sha256']
(G/'archives/legacy_finance_manifest.json').write_text(json.dumps({'archive':target.name,'archive_bytes':target.stat().st_size,'archive_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'files':rows,'excluded':excluded,'scope':'Original Finance-only SSD trees. Historical source, small data and original checkpoints. No all-SSD or other-domain model coverage claim. Selected plaintext scanned for high-confidence token patterns, zero hits; not exhaustive.'},indent=2))
print('Preserved',len(rows),'legacy Finance files;',sum(x['bytes'] for x in rows),'original bytes;',target.stat().st_size,'ZIP bytes')
