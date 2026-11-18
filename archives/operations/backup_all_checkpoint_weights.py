from pathlib import Path
import os,json,hashlib,zipfile,subprocess,time
import requests
R=Path('C:/Users/Parth/Documents/PolyvalentFinance');D=R.parent/'PolyvalentDemo';G=R.parent/'C68-FinanceRL';O=R.parent/'FinanceCheckpointRelease20261116';O.mkdir(exist_ok=True)
# Credential is read only into memory and is never printed or written to disk.
credential=subprocess.run(['git','credential','fill'],input='protocol=https\nhost=github.com\n\n',text=True,capture_output=True,cwd=G,check=True)
values=dict(x.split('=',1) for x in credential.stdout.splitlines() if '=' in x);token=values['password'];session=requests.Session();session.headers.update({'Authorization':'Bearer '+token,'Accept':'application/vnd.github+json','X-GitHub-Api-Version':'2022-11-28'})
base='https://api.github.com/repos/git-push-kar/C68-FinanceRL';tag='finance-checkpoint-archive-20261116';r=session.get(base+'/releases/tags/'+tag,timeout=60)
if r.status_code==404:
 r=session.post(base+'/releases',json={'tag_name':tag,'target_commitish':'main','name':'Finance research checkpoint archive — 16 November 2026','draft':True,'prerelease':True,'body':'Recovery archive of historical Finance RL and explanation checkpoint weights. These are research artifacts, including failed/intermediate runs, not an endorsement of every model. Metadata and full experiment context are on main under archives/ and docs/. This archive excludes upstream pristine base weights and Python environments. A verified path-to-content manifest and restore script accompany the assets. Selected packages are also available via Git LFS on main. Published only after asset size/digest verification.'},timeout=60)
r.raise_for_status();release=r.json();(O/'release.json').write_text(json.dumps({'id':release['id'],'tag':tag,'url':release['html_url']},indent=2))
index_path=O/'weights-index.json'
if not index_path.exists():
 index=[];objects={}
 for label,root in [('research',R),('demo',D)]:
  for d,dirs,files in os.walk(root):
   dirs[:]=[x for x in dirs if x not in ['.venv','.git','local_models','model_cache','__pycache__','.pytest_cache']]
   for name in files:
    p=Path(d)/name
    if p.suffix.lower() not in ['.safetensors','.pt','.bin','.pth','.ckpt','.onnx']:continue
    with p.open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
    index.append({'path':label+'/'+p.relative_to(root).as_posix(),'sha256':h,'bytes':p.stat().st_size});objects.setdefault(h,p)
 part=1;size=0;z=None;object_part={}
 for h,p in sorted(objects.items()):
  if z is None or size+p.stat().st_size>1600000000:
   if z:z.close()
   name=f'checkpoint-objects-{part:02}.zip';z=zipfile.ZipFile(O/name,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1);part+=1;size=0
  z.write(p,'objects/'+h);object_part[h]=name;size+=p.stat().st_size
 if z:z.close()
 for x in index:x['asset']=object_part[x['sha256']]
 index_path.write_text(json.dumps({'scope':'All checkpoint/weight binaries found in the research and demo workspaces, excluding upstream base/venv/cache trees. Duplicate bytes stored once. Complete paths relative to research/ and demo/.','file_count':len(index),'original_bytes':sum(x['bytes'] for x in index),'unique_objects':len(objects),'files':index},indent=2));print('Packaged',len(index),'weight paths into',part-1,'assets',flush=True)
manifest={p.name:{'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size} for p in sorted(O.glob('checkpoint-objects-*.zip'))};manifest[index_path.name]={'sha256':hashlib.sha256(index_path.read_bytes()).hexdigest(),'bytes':index_path.stat().st_size};(O/'asset-manifest.json').write_text(json.dumps(manifest,indent=2))
assets=session.get(base+f'/releases/{release["id"]}/assets',params={'per_page':100},timeout=60);assets.raise_for_status();remote={a['name']:a for a in assets.json()};receipts=[]
for p in [*sorted(O.glob('checkpoint-objects-*.zip')),index_path,O/'asset-manifest.json']:
 h=hashlib.sha256(p.read_bytes()).hexdigest();old=remote.get(p.name)
 if old:
  if old.get('digest')!='sha256:'+h or old['size']!=p.stat().st_size:raise RuntimeError('Existing release asset differs: '+p.name)
  info=old
 else:
  upload=release['upload_url'].split('{')[0]
  with p.open('rb') as f:r=session.post(upload,params={'name':p.name},data=f,headers={'Content-Type':'application/octet-stream','Content-Length':str(p.stat().st_size)},timeout=1200)
  r.raise_for_status();info=r.json()
 if info.get('digest')!='sha256:'+h or info['size']!=p.stat().st_size:raise RuntimeError('Remote asset digest/size mismatch: '+p.name)
 receipts.append({'name':p.name,'sha256':h,'bytes':p.stat().st_size,'remote_digest':info.get('digest'),'url':info['browser_download_url']});(O/'upload_receipts.json').write_text(json.dumps(receipts,indent=2));print('Verified upload',p.name,p.stat().st_size,flush=True)
# Keep draft until the restore tool has also been uploaded by the finishing step.
print('ALL WEIGHT ASSETS VERIFIED; release remains draft pending recovery tool',flush=True)
