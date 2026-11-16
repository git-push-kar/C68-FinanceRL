"""Small CPU baseline, not a replacement finance adapter."""
import hashlib,json
from pathlib import Path
import torch
from torch import nn
from summary_features import features
from model import write_json,digest

def state_digest(policy):
    h=hashlib.sha256()
    for name,value in policy.state_dict().items():
        h.update(name.encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

class NumericPolicy(nn.Module):
    def __init__(self,tickers,package=None):
        super().__init__();self.tickers=list(tickers)
        self.layers=nn.Sequential(nn.Linear(33,64),nn.Tanh(),nn.Linear(64,64),nn.Tanh(),nn.Linear(64,5))
        nn.init.zeros_(self.layers[-1].weight);nn.init.zeros_(self.layers[-1].bias)
        if package is not None:
            package=Path(package).resolve();manifest=json.loads((package/'manifest.json').read_text())
            for name,expected in manifest['sha256'].items():
                path=(package/name).resolve()
                if not path.is_relative_to(package) or digest(path)!=expected:raise ValueError('Integrity mismatch')
            meta=json.loads((package/'policy.json').read_text())
            if meta['tickers']!=self.tickers or meta['schema']!='risk_selector_summary_v1':raise ValueError('Control schema differs')
            self.load_state_dict(torch.load(package/'numeric.pt',map_location='cpu',weights_only=True))
    def forward(self,obs):
        x=torch.from_numpy(features(self.tickers,obs)).unsqueeze(0)
        logits=self.layers(x);p=.95*torch.softmax(logits/2.,-1)+.05/5
        return torch.distributions.Categorical(probs=p),logits.new_zeros(1)
    def save(self,path,metadata):
        path=Path(path);path.mkdir(parents=True,exist_ok=True)
        if (path/'manifest.json').exists():raise FileExistsError(path)
        torch.save(self.state_dict(),path/'numeric.pt')
        write_json(path/'policy.json',{'schema':'risk_selector_summary_v1','tickers':self.tickers,
            'temperature':2.,'uniform_exploration':.05,'parameters':sum(p.numel() for p in self.parameters()),'run':metadata})
        write_json(path/'manifest.json',{'sha256':{f.name:digest(f) for f in path.iterdir() if f.is_file()}})
