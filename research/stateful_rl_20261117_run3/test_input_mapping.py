import os
from pathlib import Path
import numpy as np
from transformers import AutoTokenizer
from torch import nn
from policy import NativePolicy
from market import Market

def test_shuffled_rows_keep_all_facts_and_canonical_action_mapping():
 base=os.environ.get('FINAL_BASE_DIR','C:/Users/Parth/Documents/PolyvalentFinance/local_models/InternVL3-2B')
 obj=NativePolicy.__new__(NativePolicy);nn.Module.__init__(obj)
 obj.tokenizer=AutoTokenizer.from_pretrained(base,trust_remote_code=True,local_files_only=True,use_fast=True);obj.max_tokens=0
 m=Market();t,_=m.bounds('2023-01-01','2023-12-31');obs=m.observe(t,50000.,np.zeros(5,int),50000.)
 encoded=obj.encode(obs,permutation=[4,0,3,1,2])
 for canonical_index,(ticker,position) in enumerate(zip(m.tickers,encoded['positions'])):
  decoded=obj.tokenizer.decode(encoded['ids'][0,:position+1])
  assert decoded.rstrip().endswith('Decision[ID'+str([4,0,3,1,2].index(canonical_index)+1)+']:')
  assert encoded['text'].index(ticker)<encoded['text'].index('Allocation slots:')
 assert encoded['positions']!=sorted(encoded['positions'])
 assert 'available cash 50000.00' in encoded['text'] and 'Initial capital 50000.00' in encoded['text']
 assert encoded['ids'].shape[1]<=1400
