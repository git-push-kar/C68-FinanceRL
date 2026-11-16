from pathlib import Path
import hashlib,json,subprocess,time,zipfile
import requests
G=Path('C:/Users/Parth/Documents/C68-FinanceRL');O=G.parent/'FinanceCheckpointRelease20261116'
index=json.loads((O/'weights-index.json').read_text());seen={};objects=0
for row in index['files']:seen.setdefault(row['asset'],{})[row['sha256']]=row['bytes']
for name,expected in sorted(seen.items()):
 with zipfile.ZipFile(O/name) as z:
  for h,size in expected.items():
   with z.open('objects/'+h) as f: actual=hashlib.file_digest(f,'sha256').hexdigest()
   if actual!=h or z.getinfo('objects/'+h).file_size!=size:raise ValueError('Object corruption')
   objects+=1
 print('Verified archive objects',name,len(expected),flush=True)
receipt={'unique_objects_verified':objects,'original_paths':index['file_count'],'original_bytes':index['original_bytes'],'scope':'All unique checkpoint ZIP members streamed and SHA256/size verified before publication'}
(O/'object_verification.json').write_text(json.dumps(receipt,indent=2))
while 'ALL WEIGHT ASSETS VERIFIED' not in (G.parent/'PolyvalentFinance/checkpoint-release.log').read_text():time.sleep(15)
cred=subprocess.run(['git','credential','fill'],input='protocol=https\nhost=github.com\n\n',text=True,capture_output=True,cwd=G,check=True)
token=dict(x.split('=',1) for x in cred.stdout.splitlines() if '=' in x)['password'];s=requests.Session();s.headers.update({'Authorization':'Bearer '+token,'Accept':'application/vnd.github+json','X-GitHub-Api-Version':'2022-11-28'})
api='https://api.github.com/repos/git-push-kar/C68-FinanceRL';rid=json.loads((O/'release.json').read_text())['id'];r=s.get(api+f'/releases/{rid}',timeout=60);r.raise_for_status();release=r.json()
r=s.get(api+f'/releases/{rid}/assets',params={'per_page':100},timeout=60);r.raise_for_status();remote={x['name']:x for x in r.json()}
for p in [G/'archives/restore_checkpoint_weights.py',G/'archives/CHECKPOINT_RECOVERY.md',O/'object_verification.json']:
 h=hashlib.sha256(p.read_bytes()).hexdigest();old=remote.get(p.name)
 if not old:
  with p.open('rb') as f:r=s.post(release['upload_url'].split('{')[0],params={'name':p.name},data=f,headers={'Content-Type':'application/octet-stream'},timeout=120)
  r.raise_for_status();old=r.json()
 if old['size']!=p.stat().st_size or old.get('digest')!='sha256:'+h:raise ValueError('Helper upload mismatch')
 print('Verified recovery asset',p.name,flush=True)
r=s.get(api+f'/releases/{rid}/assets',params={'per_page':100},timeout=60);r.raise_for_status();assets=r.json();remote={x['name']:x for x in assets}
for name,expected in json.loads((O/'asset-manifest.json').read_text()).items():
 a=remote[name]
 if a['size']!=expected['bytes'] or a.get('digest')!='sha256:'+expected['sha256']:raise ValueError('Final remote verification failed')
assert len(assets)==22
r=s.patch(api+f'/releases/{rid}',json={'draft':False},timeout=60);r.raise_for_status();release=r.json()
public={'release_url':release['html_url'],'tag':release['tag_name'],'draft':release['draft'],'assets':[{'name':a['name'],'bytes':a['size'],'sha256':a.get('digest'),'url':a['browser_download_url'].replace('untagged-af23840c7b7844c4042a',release['tag_name'])} for a in assets],**receipt}
(O/'published_receipt.json').write_text(json.dumps(public,indent=2));print('PUBLISHED',release['html_url'],flush=True)
