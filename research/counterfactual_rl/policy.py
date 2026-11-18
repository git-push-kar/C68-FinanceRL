"""All49 state-conditioned native InternVL3 LoRA with a numerical input adapter."""
from pathlib import Path
import hashlib,json,os,sys,time
import numpy as np
import torch
from torch import nn
from contextlib import contextmanager
from peft import LoraConfig,get_peft_model
from transformers import AutoModel,AutoTokenizer
LABELS='ABCDEFG'
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
  self.numeric_bridge=nn.Sequential(nn.Linear(25,128),nn.Tanh(),nn.Linear(128,1536,bias=False)).to(device='cuda',dtype=torch.float32)
  nn.init.zeros_(self.numeric_bridge[-1].weight)

 def encode(self,obs,permutation=None,focus=None):
  stocks=obs['stocks'];n=len(stocks)
  if n!=49 or [r['ticker'] for r in stocks]!=obs['universe']:raise ValueError('All original49 stocks required')
  order=np.arange(n) if permutation is None else np.asarray(permutation)
  if sorted(order.tolist())!=list(range(n)):raise ValueError('Permutation must include every stock once')
  # Canonical ticker order is a deterministic input contract, not learned invariance.
  stocks=sorted(stocks,key=lambda row:row['ticker']);order=np.arange(n)
  lines=['Manage all49 stocks in this portfolio. Each named stock below has one numerical state token; none are omitted.',f"As of {obs['asof']}; initialINR={obs['initial_capital']:.2f}; cashINR={obs['cash']:.2f}; equityINR={obs['equity']:.2f}.",
   'Numerical columns: quote,shares,value,weight,cash-affordability,r5,r20,r60,r120,volatility,drawdown,volume,initial capital,cash,equity,cash fraction,cap,fees,slippage,held indicator,cross-sectional rank(return20,return60,return120,volatility),quote/equity.',
   getattr(self,'action_instruction','Choose relative investment scores A=0,B=0.25,C=0.5,D=1,E=2,F=4,G=8 for each stock. Positive scores are normalized to an invested portfolio subject to a10percent stock cap. All zero means cash. Whole-share rounding minimizes portfolio tracking error including cash; no borrowing or shorts. Prices after the signal are unknown.')]
  text=self.tokenizer.apply_chat_template([{'role':'user','content':'\n'.join(lines)}],tokenize=False,add_generation_prompt=True)
  text+='\nNon-spendable corporate-action receivables: '+json.dumps(obs.get('pending_entitlements',[]),separators=(',',':'))
  numeric=[];spans={};eq=obs['equity']
  ranks={key:np.argsort(np.argsort([r[key] for r in stocks],kind='stable'),kind='stable')/(n-1) for key in ['return20','return60','return120','vol20']}
  for i in order:
   r=stocks[int(i)];text+='\nStock '+r['ticker']+' state:';spans[int(i)]=len(text)
   numeric.append([np.log1p(r['historical_quote'])/12,np.log1p(r['held_shares'])/12,r['position_value']/eq,r['weight'],r['affordable_shares_cash']*r['historical_quote']/eq,r['return5']/.1,r['return20']/.2,r['return60']/.4,r['return120']/.6,r['vol20']/.4,r['drawdown60']/.4,np.log(max(r['volume_ratio'],.01)),np.log1p(obs['initial_capital'])/12,np.log1p(obs['cash'])/12,np.log1p(eq)/12,obs['cash']/eq,obs['max_stock_weight'],obs['fees_bps']/100,obs['slippage_bps']/100,float(r['held_shares']>0)]+[float(ranks[key][int(i)]) for key in ['return20','return60','return120','vol20']]+[r['historical_quote']/eq])
  text+='\nAllocation decisions using the entire portfolio:';query={}
  for i in order:text+='\n'+stocks[int(i)]['ticker']+' action:';query[int(i)]=len(text)
  result=self.tokenizer(text,add_special_tokens=False,return_offsets_mapping=True,return_tensors='pt');offsets=result['offset_mapping'][0].tolist()
  def position(end):return next(j for j,(start,stop) in enumerate(offsets) if start<end<=stop)
  positions=[position(query[i]) for i in range(n)];state_positions=[position(spans[int(i)]) for i in order]
  ids=result['input_ids'];self.max_tokens=max(self.max_tokens,ids.shape[1])
  if ids.shape[1]>1600:raise ValueError('Full input exceeds declared budget; no truncation')
  vectors=np.clip(np.asarray(numeric,np.float32),-10,10)
  return {'ids':ids,'positions':positions,'state_positions':state_positions,'numeric_state':vectors,'canonical_numeric_state':vectors[np.argsort(order)],'prompt_sha256':hashlib.sha256(text.encode()+vectors.tobytes()).hexdigest(),'text':text,'universe':obs['universe']}

 def forward(self,encoded):
  if isinstance(encoded,list):
   if torch.is_grad_enabled():raise RuntimeError('Train focus sequences separately to preserve bounded memory')
   return torch.cat([self(e) for e in encoded],dim=0)
  base=self.model.get_base_model();lm=base.language_model;ids=encoded['ids'];emb=lm.get_input_embeddings()(ids.to(lm.get_input_embeddings().weight.device)).to(device='cuda',dtype=torch.float32)
  if not getattr(self,'ablate_numeric',False) and (not lm.lm_head.disable_adapters or getattr(self,'bridge_without_lora',False)):
   state=torch.as_tensor(encoded['numeric_state'],device='cuda',dtype=torch.float32)
   delta=.02*torch.tanh(self.numeric_bridge(state))
   emb=emb.clone();emb[0,encoded['state_positions']]=emb[0,encoded['state_positions']]+delta
   # Repeat each stock's own state at its query to prevent a fragile numerical
   # token-to-ticker binding task. Every query still attends to all49 prior states.
   own=torch.as_tensor(encoded['canonical_numeric_state'],device='cuda',dtype=torch.float32)
   emb[0,encoded['positions']]=emb[0,encoded['positions']]+.02*torch.tanh(self.numeric_bridge(own))
  hidden=lm.model(inputs_embeds=emb,attention_mask=torch.ones(ids.shape,dtype=torch.long,device='cuda'),use_cache=False,return_dict=True).last_hidden_state[0,encoded['positions']].float()
  head=lm.lm_head;logits=nn.functional.linear(hidden,self.rows)
  if not head.disable_adapters:
   a=head.lora_A['finance'];b=head.lora_B['finance'];z=a(head.lora_dropout['finance'](hidden))
   logits+=nn.functional.linear(z,b.weight[self.label_ids])*head.scaling['finance']
  return .95*torch.softmax(logits,-1)+.05/len(LABELS)

 def choose(self,obs):
  encoded=self.encode(obs);started=time.perf_counter()
  with torch.inference_mode():p=self(encoded).double().cpu().numpy()
  return p.argmax(1),{'mode':'real_all49_relative_score_model','prompt_sha256':encoded['prompt_sha256'],'tokens':encoded['ids'].shape[1],'probabilities':p.tolist(),'elapsed_seconds':time.perf_counter()-started}

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
  (path/'policy.json').write_text(json.dumps({'base':BASE_ID,'revision':REVISION,'base_weight_sha256':HASH,'labels':LABELS,'temperature':1.,'uniform_exploration':.05,'prompt_schema':'all49_canonical_relative_scores_v3','numeric_bridge':'25->128tanh->1536 per-stock input; all49 state tokens precede all49 action queries; own state repeated at query; zero initial final layer; bypassed when adapter disabled','max_tokens_observed':self.max_tokens,'trainable_parameters':sum(p.numel() for p in self.parameters() if p.requires_grad),'metadata':metadata},indent=2))
  hashes={p.relative_to(path).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in path.rglob('*') if p.is_file()}
  (path/'manifest.json').write_text(json.dumps({'sha256':hashes},indent=2))

