"""Read-only byte-level check of full/no-LoRA control initialization."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).parent
names=['strategy_ppo_internvl_seed42_256_persistent','strategy_ppo_no_lora_seed42_128']
roots=[ROOT/'campaign_runs'/n/'checkpoint_0' for n in names]
def weights(root):
    return {p.relative_to(root).as_posix():hashlib.file_digest(p.open('rb'),'sha256').hexdigest()
        for p in root.rglob('*') if p.is_file() and p.suffix in ('.pt','.safetensors') and p.name!='training_state.pt'}
left,right=map(weights,roots)
if len(left)<3 or left!=right:raise AssertionError('Initial learned-module artifacts differ')
result={'status':'passed','runs':names,'comparison':'Exact file bytes for all saved initial learned-module tensors; optimizer state and metadata excluded',
    'sha256':left,'limitations':'This proves identical saved initial modules, not equivalence of later trajectories or a financial gain.'}
out=ROOT/'reports/matched_full_frozen_lora_initialization.json'
if out.exists():raise FileExistsError(out)
out.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
