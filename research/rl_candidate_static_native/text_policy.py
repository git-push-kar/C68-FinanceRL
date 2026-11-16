"""Original InternVL text path with attention and vocabulary-projection LoRA."""
import json,os
from pathlib import Path
import torch
from torch import nn
from transformers import AutoTokenizer
from shared_backbone import SharedBackbone
from config import ModelConfig,BACKBONE_ID,BACKBONE_REVISION
from model import digest,write_json
from selected_readout import selected_readout
from exposure_prompt import prompt as exposure_prompt

class TextPolicy(nn.Module):
    def __init__(self,tickers,package=None,shared=None):
        super().__init__();self.tickers=list(tickers)
        if package is not None:
            package=Path(package).resolve();manifest=json.loads((package/'manifest.json').read_text())
            for name,expected in manifest['sha256'].items():
                path=(package/name).resolve()
                if not path.is_relative_to(package) or digest(path)!=expected:raise ValueError('Package integrity mismatch')
            record=json.loads((package/'policy.json').read_text())
            if (record['base_model_id'],record['base_revision'],record['tickers'])!=(BACKBONE_ID,BACKBONE_REVISION,self.tickers):
                raise ValueError('Base or ticker contract differs')
            if (record['temperature'],record['uniform_exploration'],record['prompt_schema'])!=(2.,.05,'candidate_static_text_v1'):
                raise ValueError('Execution schema differs')
            if record.get('adapter_scope')!='attention_readout_v1':raise ValueError('Wrong adapter/readout schema')
        self.shared=shared if shared is not None else SharedBackbone.load('cuda','fp32')
        if package is None:self.shared.add_finance_adapter(ModelConfig(dtype='fp32',lora_scope='attention_readout'))
        else:self.shared.load_adapter(Path(package)/'finance_lora/finance','finance',trainable=False)
        # Training LoRA does not require train-mode dropout. Keep identical
        # rollout/update likelihood behavior and satisfy the shared router.
        self.shared.model.eval()
        self.tokenizer=AutoTokenizer.from_pretrained(os.environ['FINANCERL_BASE_MODEL'],trust_remote_code=True,local_files_only=True)
        labels=[self.tokenizer.encode(s,add_special_tokens=False) for s in 'ABCD']
        if any(len(x)!=1 for x in labels):raise ValueError('Action labels must be single vocabulary tokens')
        self.label_ids=[x[0] for x in labels]
        if package is not None and record['labels']!=self.label_ids:raise ValueError('Action-token identities differ')
        head=self.shared.base.language_model.lm_head
        if head.get_base_layer().bias is not None:raise ValueError('Unexpected original vocabulary bias')
        head.lora_A['finance'].to(device=self.shared.compute_device,dtype=torch.float32)
        head.lora_B['finance'].to(device=self.shared.compute_device,dtype=torch.float32)
        self.register_buffer('readout',head.get_base_layer().weight[self.label_ids].detach().float().cuda())
        self.max_observed_tokens=0
    def prompt(self,obs):
        return exposure_prompt(obs,self.tickers)
    def action_logits(self,obs):
        if obs['summary'].shape[0]!=1:raise ValueError('Use exact single-sample likelihood kernels')
        prompt=self.prompt(obs)
        ids=self.tokenizer.apply_chat_template([{'role':'user','content':prompt}],tokenize=True,add_generation_prompt=True,return_tensors='pt')
        self.max_observed_tokens=max(self.max_observed_tokens,ids.shape[1])
        if ids.shape[1]>1024:raise ValueError('Prompt exceeds declared 1024-token complete-input budget; do not silently truncate')
        embedding_layer=self.shared.base.language_model.get_input_embeddings()
        embeddings=embedding_layer(ids.to(embedding_layer.weight.device)).to(device=self.shared.compute_device,dtype=torch.float32)
        h=self.shared.finance_hidden_states(embeddings)[:,-1].float()
        logits=selected_readout(h,self.shared.base.language_model.lm_head,self.readout,self.label_ids)
        return logits
    def forward(self,obs):
        logits=self.action_logits(obs)
        # Fixed exploration temperature, identical in the base and tuned policies.
        probabilities=.95*torch.softmax(logits/2.,-1)+.05/4
        return torch.distributions.Categorical(probs=probabilities),logits.new_zeros(1)
    def save(self,path,metadata):
        path=Path(path);path.mkdir(parents=True,exist_ok=True)
        if (path/'manifest.json').exists():raise FileExistsError('Do not overwrite a saved policy')
        self.shared.model.save_pretrained(path/'finance_lora',selected_adapters=['finance'],save_embedding_layers=False)
        write_json(path/'policy.json',{'base_model_id':BACKBONE_ID,'base_revision':BACKBONE_REVISION,
            'tickers':self.tickers,'labels':self.label_ids,'temperature':2.,'uniform_exploration':.05,
            'only_trainable_component':'rank16 attention and original-vocabulary-projection LoRA',
            'adapter_scope':'attention_readout_v1','prompt_schema':'candidate_static_text_v1',
            'max_observed_tokens':self.max_observed_tokens,'run':metadata})
        write_json(path/'manifest.json',{'sha256':{f.relative_to(path).as_posix():digest(f) for f in path.rglob('*') if f.is_file()}})
