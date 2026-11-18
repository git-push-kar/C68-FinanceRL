import gc,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from text_policy import TextPolicy
from candidate_market import Market,tensor
from train import lora_digest,seed_everything
from model import write_json
from execution_policy import configure_finance_precision
from selected_readout import selected_readout

def frozen_hash(policy):
    h=hashlib.sha256()
    for name,param in policy.named_parameters():
        if 'lora_' not in name:
            h.update(name.encode());h.update(param.detach().float().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def main():
    out=Path('../checks/risk_selector_native_gate');out.mkdir(parents=True,exist_ok=False)
    seed_everything(42);configure_finance_precision();torch.set_num_threads(2)
    market=Market('../multiasset_data_20261115/five_asset_development_through2024.csv','2010-01-01','2010-12-31')
    obs,_=market.reset();policy=TextPolicy(market.env.tickers);input=tensor(obs,'cuda')
    names=[n for n,p in policy.named_parameters() if p.requires_grad]
    if not names or not all('lora_' in n for n in names):raise AssertionError('Non-LoRA trainable parameter')
    before=frozen_hash(policy);lora_before=lora_digest(policy)
    with torch.no_grad():expected=policy(input)[0].probs.detach()
    initial_probabilities=expected[0].cpu().tolist()
    with torch.no_grad(),policy.shared.model.disable_adapter():
        original=policy(input)[0].probs[0].cpu().numpy()
    base_reference_error=float(np.max(abs(np.array(initial_probabilities)-original)))
    if base_reference_error>1e-7:raise AssertionError('Initial adapter changed original exposure policy')
    trainable_count=sum(p.numel() for p in policy.parameters() if p.requires_grad)
    actual=policy(input)[0].probs
    discrepancy=float((expected-actual.detach()).abs().max())
    if discrepancy>1e-7:raise AssertionError('Rollout/update mismatch')
    optimizer=torch.optim.AdamW([
        {'params':[p for n,p in policy.named_parameters() if p.requires_grad and '.lm_head.' not in n],'lr':1e-5},
        {'params':[p for n,p in policy.named_parameters() if p.requires_grad and '.lm_head.' in n],'lr':3e-4}],weight_decay=0)
    started=time.monotonic();loss=-actual[0,0].log();loss.backward()
    grad=float(torch.nn.utils.clip_grad_norm_([p for p in policy.parameters() if p.requires_grad],1.,error_if_nonfinite=True))
    optimizer.step();after=frozen_hash(policy)
    if before!=after or lora_before==lora_digest(policy):raise AssertionError('Frozen-base or adapter update failure')
    with torch.no_grad():prediction=policy(input)[0].probs[0].cpu().numpy()
    training_peak=torch.cuda.max_memory_allocated()/2**30
    # Check the actual model's complete vocabulary head, not merely a toy.
    with torch.no_grad():
        head=policy.shared.base.language_model.lm_head
        dense=head.get_base_layer().weight.detach().float().cuda()
        probe=torch.randn(1,policy.shared.hidden_size,device='cuda')
        full=torch.nn.functional.linear(probe,dense)+head.lora_B['finance'](head.lora_A['finance'](probe))*head.scaling['finance']
        partial=selected_readout(probe,head,policy.readout,policy.label_ids)
        full_head_error=float((partial-full[:,policy.label_ids]).abs().max())
        torch.testing.assert_close(partial,full[:,policy.label_ids],atol=1e-5,rtol=1e-5)
        del dense,probe,full,partial,head
    tokens=policy.max_observed_tokens;peak=torch.cuda.max_memory_allocated()/2**30
    policy.save(out/'package',{'role':'hardware gate, not financial training result'})
    del optimizer,actual,loss,policy;gc.collect();torch.cuda.empty_cache()
    reloaded=TextPolicy(market.env.tickers,out/'package')
    with torch.no_grad():restored=reloaded(input)[0].probs[0].cpu().numpy()
    error=float(np.max(abs(prediction-restored)))
    if error>1e-7:raise AssertionError('Reload mismatch')
    result={'status':'passed','only_LoRA_trainable':True,'frozen_before':before,'frozen_after':after,
        'trainable_parameters':trainable_count,'initial_action_probabilities':initial_probabilities,
        'original_native_policy_initial_probability_error':base_reference_error,
        'after_one_technical_update_probabilities':prediction.tolist(),
        'initial_probability_discrepancy':discrepancy,'reload_max_probability_error':error,
        'gradient_norm':grad,'complete_prompt_tokens':tokens,'peak_allocated_GiB':peak,
        'training_peak_allocated_GiB':training_peak,'full_vocabulary_selected_logits_max_error':full_head_error,
        'seconds':time.monotonic()-started,'financial_performance_tested':False}
    write_json(out/'result.json',result);print(json.dumps(result,indent=2))

if __name__=='__main__':main()
