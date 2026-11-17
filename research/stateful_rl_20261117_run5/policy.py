"""State-conditioned native InternVL3 LoRA: five stock decisions, seven actions.

No learned numerical substitute, cached decision, fixed-mixture guard, or time gate.
"""
from pathlib import Path
import hashlib,json,os,sys,time
import numpy as np
import numpy as np
import torch
from torch import nn
from contextlib import contextmanager
from peft import LoraConfig,get_peft_model
from transformers import AutoModel,AutoTokenizer
from market import prompt,LABELS
from numeric_state import numeric_features
from safetensors.torch import save_file

BASE_ID='OpenGVLab/InternVL3-2B'
REVISION='899155015275a9b7338c7f4677e19c784e0e5a21'
HASH='18b4cdbc8eab790f447feac717f4d0cecbb85caf8df198b3b2401c22506858e7'
TARGETS=r'(language_model\.model\.layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj|o_proj)|language_model\.lm_head)'

def checkpoint_layers(decoder):
 from torch.utils.checkpoint import checkpoint
 for layer in decoder.layers:
  original=layer.forward
  def wrap(forward):
   def call(*args,**kwargs):
    if torch.is_grad_enabled():return checkpoint(forward,*args,use_reentrant=False,preserve_rng_state=True,**kwargs)
    return forward(*args,**kwargs)
   return call
  layer.forward=wrap(original)

class NativePolicy(nn.Module):
 def __init__(self,base_path):
  super().__init__();base_path=Path(base_path)
  with (base_path/'model.safetensors').open('rb') as f:
   if hashlib.file_digest(f,'sha256').hexdigest()!=HASH:raise ValueError('Wrong pristine base weights')
  torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
  raw=AutoModel.from_pretrained(str(base_path),trust_remote_code=True,torch_dtype=torch.bfloat16,low_cpu_mem_usage=True,use_flash_attn=False,local_files_only=True).requires_grad_(False).eval()
  if (raw.config.model_type,raw.config.llm_config.hidden_size,raw.config.llm_config.num_hidden_layers)!=('internvl_chat',1536,28):raise ValueError('Wrong backbone architecture')
  cfg=LoraConfig(r=16,lora_alpha=32,lora_dropout=0.,target_modules=TARGETS,bias='none',revision=REVISION)
  self.model=get_peft_model(raw,cfg,adapter_name='finance');self.model.peft_config['finance'].base_model_name_or_path=BASE_ID;self.model.peft_config['finance'].revision=REVISION
  base=self.model.get_base_model();decoder=base.language_model.model
  decoder.layers.to(device='cuda',dtype=torch.float32);decoder.norm.to(device='cuda',dtype=torch.float32)
  if hasattr(decoder,'rotary_emb'):decoder.rotary_emb.to(device='cuda',dtype=torch.float32)
  checkpoint_layers(decoder)
  self.tokenizer=AutoTokenizer.from_pretrained(str(base_path),trust_remote_code=True,local_files_only=True,use_fast=True)
  label_ids=[self.tokenizer.encode(x,add_special_tokens=False) for x in LABELS]
  if any(len(x)!=1 for x in label_ids):raise ValueError('Action labels must have one token')
  self.label_ids=[x[0] for x in label_ids]
  head=base.language_model.lm_head
  if head.get_base_layer().bias is not None:raise ValueError('Unexpected vocabulary bias')
  head.lora_A['finance'].to(device='cuda',dtype=torch.float32);head.lora_B['finance'].to(device='cuda',dtype=torch.float32)
  self.register_buffer('rows',head.get_base_layer().weight[self.label_ids].detach().float().cuda())
  self.max_tokens=0;self.model.eval()
  if any('lora_' not in name for name,p in self.named_parameters() if p.requires_grad):raise ValueError('Trainable base or external head')
  # Numerical STATE encoder, not an action head: its output enters the input
  # embedding of the original28-layer LM. Only native vocabulary logits act.
  self.numeric_bridge=nn.Sequential(nn.Linear(38,128),nn.Tanh(),nn.Linear(128,1536,bias=False)).to(device='cuda',dtype=torch.float32)
  nn.init.zeros_(self.numeric_bridge[-1].weight)

 def encode(self,obs,permutation=None,focus=None):
  if focus is None:return [self.encode(obs,permutation,i) for i in range(5)]
  stocks=obs['stocks'];s=stocks[int(focus)];order=range(5) if permutation is None else permutation
  lines=['Indian whole-share portfolio. Choose ONE action for the FOCUS stock using current facts; other allocations are decided separately.',
   'A=KEEP shares; B=0%; C=5%; D=10%; E=20%; F=30%; G=maximum permitted stock weight. Unallocated capital stays cash. Combined weights above100% are normalized.',
   'No borrowing/shorts. Sell before buying. Execute next unknown open; fees0.10% and slippage0.05% each side. Reassess after20sessions. Objective net wealth growth with controlled drawdown.',
   f"Date={obs['asof']} initial_INR={obs['initial']:.0f} equity_INR={obs['equity']:.2f} cash_INR={obs['cash']:.2f}; stock_cap={obs['max_stock_weight']:.2f} sector_cap={obs['max_sector_weight']:.2f}; limits override KEEP.",
   'Portfolio rows: stock/business,priceINR,shares,weight,return20/60/120,vol20. All indicators are historical.']
  for i in order:
   r=stocks[int(i)];lines.append(f"{r['ticker']} {r['company']}: price={r['price']:.2f} shares={r['shares']} weight={r['weight']:.3f} r20={r['return20']:.3f} r60={r['return60']:.3f} r120={r['return120']:.3f} vol={r['vol20']:.3f}")
  relative=s['vol20']/max(float(np.mean([r['vol20'] for r in stocks])),1e-8)
  lines.extend([f"FOCUS STOCK: {s['ticker']} ({s['company']}). PriceINR={s['price']:.2f}; held_shares={s['shares']}; weight={s['weight']:.3f}; affordable_shares_from_cash={s['affordable_shares_at_signal']}; affordable_from_total_equity={'YES' if s['price']*1.0015005<=obs['equity'] else 'NO'}.",
   f"FOCUS signals: return5={s['return5']:.3f}; return20={s['return20']:.3f}; return60={s['return60']:.3f}; return120={s['return120']:.3f}; drawdown60={s['drawdown60']:.3f}; annual_vol20={s['vol20']:.3f}; volume_ratio={s['volume_ratio']:.2f}; volatility_relative_to_portfolio_mean={relative:.2f}.",
   'Choose the action letter for this FOCUS stock.'])
  text=self.tokenizer.apply_chat_template([{'role':'user','content':'\n'.join(lines)}],tokenize=False,add_generation_prompt=True)
  text+='Action:';ids=self.tokenizer(text,add_special_tokens=False,return_tensors='pt')['input_ids'];self.max_tokens=max(self.max_tokens,ids.shape[1])
  if ids.shape[1]>1400:raise ValueError('Complete input exceeds 1400 tokens; truncation forbidden')
  vector=numeric_features(obs,int(focus))
  digest=hashlib.sha256(text.encode()+vector.tobytes()).hexdigest()
  return {'ids':ids,'positions':[ids.shape[1]-1],'focus':int(focus),'numeric_state':vector,'prompt_sha256':digest,'text':text}

 def forward(self,encoded):
  if isinstance(encoded,list):
   if torch.is_grad_enabled():raise RuntimeError('Train focus sequences separately to preserve bounded memory')
   return torch.cat([self(e) for e in encoded],dim=0)
  base=self.model.get_base_model();lm=base.language_model;ids=encoded['ids'];emb=lm.get_input_embeddings()(ids.to(lm.get_input_embeddings().weight.device)).to(device='cuda',dtype=torch.float32)
  if not lm.lm_head.disable_adapters:
   state=torch.as_tensor(encoded['numeric_state'],device='cuda',dtype=torch.float32)
   delta=.02*torch.tanh(self.numeric_bridge(state))
   emb=emb.clone();emb[0,-1]=emb[0,-1]+delta
  hidden=lm.model(inputs_embeds=emb,attention_mask=torch.ones(ids.shape,dtype=torch.long,device='cuda'),use_cache=False,return_dict=True).last_hidden_state[0,encoded['positions']].float()
  head=lm.lm_head;logits=nn.functional.linear(hidden,self.rows)
  if not head.disable_adapters:
   a=head.lora_A['finance'];b=head.lora_B['finance'];z=a(head.lora_dropout['finance'](hidden))
   logits+=nn.functional.linear(z,b.weight[self.label_ids])*head.scaling['finance']
  return .95*torch.softmax(logits,-1)+.05/len(LABELS)

 def choose(self,obs):
  encoded=self.encode(obs);started=time.perf_counter()
  with torch.inference_mode():p=self(encoded).double().cpu().numpy()
  return p.argmax(1),{'mode':'real_focused_stock_model','prompt_sha256':[e['prompt_sha256'] for e in encoded],'tokens':max(e['ids'].shape[1] for e in encoded),'probabilities':p.tolist(),'elapsed_seconds':time.perf_counter()-started}

 def freeze_reference(self):
  self.reference_weights=[p.detach().clone() for p in self.parameters() if p.requires_grad]

 @contextmanager
 def reference_context(self):
  if not hasattr(self,'reference_weights'):
   with self.model.disable_adapter():yield
   return
  parameters=[p for p in self.parameters() if p.requires_grad]
  current=[p.detach().clone() for p in parameters]
  try:
   with torch.no_grad():
    for p,r in zip(parameters,self.reference_weights):p.copy_(r)
   yield
  finally:
   with torch.no_grad():
    for p,r in zip(parameters,current):p.copy_(r)

 def save(self,path,metadata):
  path=Path(path);path.mkdir(parents=True,exist_ok=False)
  self.model.save_pretrained(path/'finance_lora',selected_adapters=['finance'],save_embedding_layers=False,safe_serialization=True)
  save_file({k:v.detach().cpu().contiguous() for k,v in self.numeric_bridge.state_dict().items()},str(path/'numeric_bridge.safetensors'))
  (path/'policy.json').write_text(json.dumps({'base':BASE_ID,'revision':REVISION,'base_weight_sha256':HASH,'labels':LABELS,'temperature':1.,'uniform_exploration':.05,'prompt_schema':'focused_stock_numeric_input_bridge_v5','numeric_bridge':'38->128 tanh->1536;0.02tanh residual at last query input embedding; zero initial last layer; bypassed when adapter disabled','max_tokens_observed':self.max_tokens,'trainable_parameters':sum(p.numel() for p in self.parameters() if p.requires_grad),'metadata':metadata},indent=2))
  hashes={p.relative_to(path).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in path.rglob('*') if p.is_file()}
  (path/'manifest.json').write_text(json.dumps({'sha256':hashes},indent=2))

