"""CPU-only native input audit; does not construct or train the backbone."""
import argparse,json,os,sys
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer
ROOT=Path(__file__).parent
parser=argparse.ArgumentParser();parser.add_argument('--branch',choices=['rl_exposure_native_trial','rl_exposure_direct_native_trial'],default='rl_exposure_native_trial');args=parser.parse_args()
sys.path.insert(0,str(ROOT/args.branch))
from exposure_market import Market,tensor
from exposure_prompt import prompt,LABELS
data=ROOT/'sector_data_20261107/sector9_development_through2024.csv'
tokenizer=AutoTokenizer.from_pretrained(ROOT/'local_models/InternVL3-2B',trust_remote_code=True,local_files_only=True)
rows=[]
for split,start,end in [('training','2001-01-01','2018-12-31'),('validation','2019-01-01','2021-12-31'),('development','2022-01-01','2024-12-31')]:
    market=Market(data,start,end)
    indices=np.linspace(market.env.start_index,len(market.env.prices)-7,20,dtype=int)
    for index in indices:
        obs,_=market.reset(int(index));text=prompt(tensor(obs,'cpu'),market.env.tickers)
        for label,value in zip(LABELS,obs['summary'][:12]):
            if f'{label}: {value:.2f}%' not in text:raise AssertionError('Input mismatch')
        for label,value in zip('ABCD',obs['summary'][12:]):
            if f'{label}={value:.2f}' not in text:raise AssertionError('Cost mismatch')
        ids=tokenizer.apply_chat_template([{'role':'user','content':text}],tokenize=True,add_generation_prompt=True)
        if len(ids)>1024:raise AssertionError('Complete prompt exceeds budget')
        rows.append({'split':split,'execution_date':str(market.env.dates[index])[:10],'tokens':len(ids)})
result={'status':'passed','scope':'CPU tokenizer and input audit only; no model inference or financial test',
        'samples':rows,'minimum_tokens':min(x['tokens'] for x in rows),'maximum_tokens':max(x['tokens'] for x in rows),
        'displayed_numeric_features':16,'reserved_test_accessed':False}
out=ROOT/'checks'/('exposure_native_prompt_audit.json' if args.branch=='rl_exposure_native_trial' else 'exposure_direct_native_prompt_audit.json')
if out.exists():raise FileExistsError(out)
out.write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='samples'},indent=2))
