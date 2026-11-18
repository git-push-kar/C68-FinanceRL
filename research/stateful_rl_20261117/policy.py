"""State-conditioned native InternVL3 LoRA: five stock decisions, seven actions.

No learned numerical substitute, cached decision, fixed-mixture guard, or time gate.
"""
from pathlib import Path
import hashlib,json,os,sys,time
import numpy as np
import torch
from torch import nn
from peft import LoraConfig,get_peft_model
from transformers import AutoModel,AutoTokenizer
from market import prompt,LABELS

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

 def encode(self,obs):
  text=self.tokenizer.apply_chat_template([{'role':'user','content':prompt(obs)}],tokenize=False,add_generation_prompt=True)
  markers=['Decision['+s['ticker']+']:' for s in obs['stocks']]
  text+='\nAllocation slots:\n'+'\n'.join(markers)
  encoded=self.tokenizer(text,add_special_tokens=False,return_offsets_mapping=True,return_tensors='pt')
  offsets=encoded['offset_mapping'][0].tolist();positions=[]
  for marker in markers:
   end=text.index(marker)+len(marker);positions.append(max(i for i,(_,b) in enumerate(offsets) if b<=end and b>0))
  ids=encoded['input_ids'];self.max_tokens=max(self.max_tokens,ids.shape[1])
  if ids.shape[1]>1400:raise ValueError('Complete input exceeds 1400 tokens; truncation forbidden')
  return {'ids':ids,'positions':positions,'prompt_sha256':hashlib.sha256(text.encode()).hexdigest(),'text':text}

 def forward(self,encoded):
  base=self.model.get_base_model();lm=base.language_model;ids=encoded['ids'];emb=lm.get_input_embeddings()(ids.to(lm.get_input_embeddings().weight.device)).to(device='cuda',dtype=torch.float32)
  hidden=lm.model(inputs_embeds=emb,attention_mask=torch.ones(ids.shape,dtype=torch.long,device='cuda'),use_cache=False,return_dict=True).last_hidden_state[0,encoded['positions']].float()
  head=lm.lm_head;logits=nn.functional.linear(hidden,self.rows)
  if not head.disable_adapters:
   a=head.lora_A['finance'];b=head.lora_B['finance'];z=a(head.lora_dropout['finance'](hidden))
   logits+=nn.functional.linear(z,b.weight[self.label_ids])*head.scaling['finance']
  return .95*torch.softmax(logits/1.5,-1)+.05/len(LABELS)

 def choose(self,obs):
  encoded=self.encode(obs);started=time.perf_counter()
  with torch.inference_mode():p=self(encoded).double().cpu().numpy()
  return p.argmax(1),{'mode':'real_state_conditioned_model','prompt_sha256':encoded['prompt_sha256'],'tokens':encoded['ids'].shape[1],'probabilities':p.tolist(),'elapsed_seconds':time.perf_counter()-started}

 def save(self,path,metadata):
  path=Path(path);path.mkdir(parents=True,exist_ok=False)
  self.model.save_pretrained(path/'finance_lora',selected_adapters=['finance'],save_embedding_layers=False,safe_serialization=True)
  (path/'policy.json').write_text(json.dumps({'base':BASE_ID,'revision':REVISION,'base_weight_sha256':HASH,'labels':LABELS,'temperature':1.5,'uniform_exploration':.05,'prompt_schema':'five_stock_full_account_v1','max_tokens_observed':self.max_tokens,'trainable_parameters':sum(p.numel() for p in self.parameters() if p.requires_grad),'metadata':metadata},indent=2))
  hashes={p.relative_to(path).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in path.rglob('*') if p.is_file()}
  (path/'manifest.json').write_text(json.dumps({'sha256':hashes},indent=2))

def expected_reward(probabilities,rewards):
 """Exact finite-action expectation, equivalent to all-action policy gradient.

 All 7^5 combinations are evaluated by the historical simulator; there is no
 teacher answer or hindsight action in the observation. Gradient flows only
 through the five original-vocabulary action distributions.
 """
 joint=probabilities[0]
 for row in probabilities[1:]:joint=joint.unsqueeze(-1)*row
 return (joint*rewards.reshape((7,)*5)).sum()
