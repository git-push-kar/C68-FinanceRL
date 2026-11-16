"""Exact selected-vocabulary-row evaluation of an unmerged standard LoRA head."""
import torch.nn.functional as F

def selected_readout(hidden,head,base_rows,label_ids):
    if head.merged:raise ValueError('Cached original rows require unmerged LoRA')
    result=F.linear(hidden,base_rows)
    if head.disable_adapters:return result
    if list(head.active_adapters)!=['finance']:raise ValueError('Activate exactly the finance adapter')
    if head.use_dora['finance']:raise ValueError('This schema supports standard LoRA only')
    a=head.lora_A['finance'];b=head.lora_B['finance']
    z=a(head.lora_dropout['finance'](hidden.to(a.weight.dtype)))
    bias=None if b.bias is None else b.bias[label_ids]
    return result+F.linear(z,b.weight[label_ids],bias)*head.scaling['finance']
