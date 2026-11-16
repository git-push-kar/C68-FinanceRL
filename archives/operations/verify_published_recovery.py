from pathlib import Path
import hashlib,json,requests,shutil
G=Path('C:/Users/Parth/Documents/C68-FinanceRL');O=G.parent/'FinanceCheckpointRelease20261116'
url='https://api.github.com/repos/git-push-kar/C68-FinanceRL/releases/tags/finance-checkpoint-archive-20261116'
r=requests.get(url,timeout=60);r.raise_for_status();release=r.json();assert not release['draft']
remote={a['name']:a for a in release['assets']};assert len(remote)==22
for name,meta in json.loads((O/'asset-manifest.json').read_text()).items():
 a=remote[name];assert a['size']==meta['bytes'] and a['digest']=='sha256:'+meta['sha256']
for name in ['asset-manifest.json','weights-index.json','restore_checkpoint_weights.py']:
 r=requests.get(remote[name]['browser_download_url'],timeout=60);r.raise_for_status();b=r.content
 expected=hashlib.sha256(((G/'archives'/name) if name.endswith('.py') else (O/name)).read_bytes()).hexdigest()
 assert hashlib.sha256(b).hexdigest()==expected
 print('Anonymous public download verified',name,len(b),flush=True)
receipt=json.loads((O/'published_receipt.json').read_text());receipt['public_api_verified']=True;receipt['public_downloads_verified']=['asset-manifest.json','weights-index.json','restore_checkpoint_weights.py'];receipt['asset_count']=len(remote);receipt['compressed_asset_bytes']=sum(a['size'] for a in remote.values())
for a in receipt['assets']:a['url']=remote[a['name']]['browser_download_url']
(G/'archives/published_checkpoint_receipt.json').write_text(json.dumps(receipt,indent=2))
print('Public release verified',len(remote),'assets;',receipt['compressed_asset_bytes'],'bytes')
