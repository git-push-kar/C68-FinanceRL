"""Small numeric control; same causal inputs and curriculum, no language model.

Does not inspect the financial test. Kept outside the frozen worker source tree.
"""
import argparse,json,sys
from pathlib import Path
import numpy as np
import torch
from torch import nn
from transformers import AutoTokenizer
sys.path.insert(0,str(Path(__file__).parent/'full_universe_rl'))
from policy import NativePolicy
from market import Market,evaluate
from train import write

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);args=ap.parse_args();out=Path(args.run);torch.set_num_threads(2);torch.manual_seed(42)
 encoder=NativePolicy.__new__(NativePolicy);nn.Module.__init__(encoder);encoder.max_tokens=0
 encoder.tokenizer=AutoTokenizer.from_pretrained('C:/Users/Parth/Documents/PolyvalentFinance/local_models/InternVL3-2B',trust_remote_code=True,local_files_only=True,use_fast=True)
 def features(obs):
  x=encoder.encode(obs)['canonical_numeric_state'];return np.concatenate([x,np.broadcast_to(x.mean(0),x.shape),np.broadcast_to(x.std(0),x.shape)],axis=1)
 def tensors(rows):return torch.tensor(np.concatenate([features(r['observation']) for r in rows])),torch.tensor(np.array([r['actions'] for r in rows]).ravel())
 rows=json.loads((out/'curriculum.json').read_text());gate=json.loads((out/'gate_examples.json').read_text());x,y=tensors(rows);gx,gy=tensors(gate);mean=x.mean(0);std=x.std(0).clamp(min=.05);x=(x-mean)/std;gx=(gx-mean)/std
 model=nn.Sequential(nn.Linear(60,128),nn.Tanh(),nn.Linear(128,64),nn.Tanh(),nn.Linear(64,7));opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
 count=torch.bincount(y,minlength=7);weights=1/count.clamp(min=1).float().sqrt();weights/=weights.mean();history=[]
 def score(a,b):
  with torch.no_grad():pred=model(a).argmax(1)
  counts=torch.bincount(b,minlength=7);correct=torch.bincount(b[b==pred],minlength=7);recall=correct/counts.clamp(min=1);eligible=counts>=15
  return {'accuracy':float((pred==b).float().mean()),'balanced_accuracy':float(recall[eligible].mean()),'recall':recall.tolist()}
 for epoch in range(65):
  if epoch%16==0:history.append({'epoch':epoch,'train':score(x,y),'gate':score(gx,gy)});write(out/'numeric_control_progress.json',history)
  if epoch==64:break
  for ix in torch.randperm(len(x)).split(1024):
   opt.zero_grad();loss=nn.functional.cross_entropy(model(x[ix]),y[ix],weight=weights,label_smoothing=.05);loss.backward();opt.step()
 m=Market()
 def choose(obs):
  with torch.no_grad():p=model((torch.tensor(features(obs))-mean)/std).softmax(-1).numpy()
  return p.argmax(1),{'mode':'numeric_control','probabilities':p.tolist()}
 result=evaluate(m,choose,'2024-01-01','2024-12-31',100000);write(out/'numeric_control_validation.json',result)
 torch.save({'model':model.state_dict(),'mean':mean,'std':std},out/'numeric_control.pt')
 write(out/'numeric_control_summary.json',{'scope':'Same supervised labels and causal20-per-stock inputs, plus permutation-invariant mean/std context; smallMLP, no LM, no RL, no financial test access. Fixed64epochs. This is a representation/SFT control, not a replacement for the requested2B RL model.','parameters':sum(p.numel() for p in model.parameters()),'history':history,'validation':result['metrics']});print(json.dumps(history[-1]),flush=True)

if __name__=='__main__':main()
