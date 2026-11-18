import numpy as np
from torch import nn
from transformers import AutoTokenizer
from policy import NativePolicy
from market import Market
from learning_gate import passes

def test_focus_is_explicit_and_full_account_preserved():
 obj=NativePolicy.__new__(NativePolicy);nn.Module.__init__(obj);obj.max_tokens=0
 obj.tokenizer=AutoTokenizer.from_pretrained('C:/Users/Parth/Documents/PolyvalentFinance/local_models/InternVL3-2B',trust_remote_code=True,local_files_only=True,use_fast=True)
 m=Market();t,_=m.bounds('2023-01-01','2023-06-01');o=m.observe(t,500.,np.array([0,1,2,3,4]),50000.)
 encoded=obj.encode(o,permutation=[4,3,2,1,0]);assert len(encoded)==5
 for i,e in enumerate(encoded):
  assert e['focus']==i and 'FOCUS STOCK: '+m.tickers[i] in e['text']
  assert all(ticker in e['text'] for ticker in m.tickers)
  assert 'cash_INR=500.00' in e['text'] and 'initial_INR=50000' in e['text']
  assert e['positions']==[e['ids'].shape[1]-1] and e['text'].endswith('Action:')
 assert len({e['prompt_sha256'] for e in encoded})==5

def test_constant_answer_cannot_pass_learning_gate():
 result={'balanced_accuracy':1/7,'minimum_eligible_recall':0.,'distinct_predictions':1}
 assert not passes(result,result)
