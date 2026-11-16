"""Check complete prompts without loading the backbone or inspecting test data."""
import json,os
from pathlib import Path
import numpy as np
from torch import nn
from transformers import AutoTokenizer
from text_policy import TextPolicy
from strategy_ppo import Market,tensor

def main():
    market=Market('../sector_data_20261107/sector9_development_through2024.csv','2001-01-01','2024-12-31')
    formatter=TextPolicy.__new__(TextPolicy);nn.Module.__init__(formatter);formatter.tickers=market.env.tickers
    tokenizer=AutoTokenizer.from_pretrained(os.environ['FINANCERL_BASE_MODEL'],trust_remote_code=True,local_files_only=True)
    labels=[tokenizer.encode(s,add_special_tokens=False) for s in 'ABCD']
    assert all(len(x)==1 for x in labels)
    state=market.reset();lengths=[]
    while True:
        obs,targets=state
        prompt=formatter.prompt(tensor(obs,'cpu'))
        ids=tokenizer.apply_chat_template([{'role':'user','content':prompt}],tokenize=True,add_generation_prompt=True)
        lengths.append(len(ids))
        assert len(ids)<=1024,'Never truncate observations'
        assert all(t+':' in prompt for t in market.env.tickers)
        assert 'nan' not in prompt.lower() and 'inf,' not in prompt.lower()
        state,_,done=market.step(np.ones(4)/4@targets)
        if done:break
    result={'status':'passed','scope':'Complete tokenization under uniform-mixture holdings, development only; no model loaded',
        'prompt_count':len(lengths),'minimum_tokens':min(lengths),'maximum_tokens':max(lengths),
        'action_label_ids':labels,'declared_limit':1024,'truncation':False,'future_test_accessed':False}
    out=Path('../checks/native_text_input_audit.json')
    out.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

if __name__=='__main__':main()
