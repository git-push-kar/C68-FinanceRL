import numpy as np
from torch import nn
from transformers import AutoTokenizer
from policy import NativePolicy
from state import OriginalPanel

def test_every_stock_state_is_before_every_action_and_no_rows_lost():
 obj=NativePolicy.__new__(NativePolicy);nn.Module.__init__(obj);obj.max_tokens=0
 obj.tokenizer=AutoTokenizer.from_pretrained('C:/Users/Parth/Documents/PolyvalentFinance/local_models/InternVL3-2B',trust_remote_code=True,local_files_only=True,use_fast=True)
 p=OriginalPanel();t=int(np.searchsorted(p.dates,np.datetime64('2024-01-02')));o=p.observe(t,10000.,np.arange(49,dtype=np.int64),100000.,np.linspace(100,5000,49))
 e=obj.encode(o,permutation=np.arange(49)[::-1]);assert e['numeric_state'].shape==(49,20)
 assert len(e['positions'])==len(e['state_positions'])==49
 assert max(e['state_positions'])<min(e['positions'])
 for i,position in enumerate(e['positions']):
  assert obj.tokenizer.decode(e['ids'][0,:position+1]).rstrip().endswith(p.tickers[i]+' action:')
 assert all(ticker in e['text'] for ticker in p.tickers)
 assert e['ids'].shape[1]<=1600
 for number,i in enumerate(np.arange(49)[::-1]):assert e['numeric_state'][number,3]==np.float32(o['stocks'][i]['weight'])
