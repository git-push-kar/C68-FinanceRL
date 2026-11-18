import argparse,json,sys,os
from pathlib import Path
R=Path(__file__).parent;a=argparse.ArgumentParser();a.add_argument('--run',required=True);r=Path(a.parse_args().run).resolve()
sys.path.insert(0,str(R/'rl_static_chronology'));os.chdir(R/'rl_static_chronology')
import torch
import numpy as np
from text_policy import TextPolicy
from candidate_market import Market,tensor,evaluate
from execution_policy import configure_finance_precision,wait_for_compute_window
wait_for_compute_window();configure_finance_precision();torch.set_num_threads(2)
meta=json.loads((r/'protocol.json').read_text());c=max(json.loads((r/'candidates.json').read_text()),key=lambda x:x['step'])
m=Market('../data/nifty50_prices.csv',*meta['development']);p=TextPolicy(m.env.tickers,c['package'])
with torch.no_grad():v=p(tensor(m.reset()[0],'cuda'))[0].probs[0].double();v=(v/v.sum()).cpu().numpy()
actual=evaluate(None,m,'cpu',fixed=v);expected=json.loads((r/f'development_{c["step"]}.json').read_text())
error=float(np.max(abs(np.asarray(actual['probabilities'])-np.asarray(expected['probabilities']))))
daily=float(np.max(abs(np.asarray([x['net_return'] for x in actual['daily']])-np.asarray([x['net_return'] for x in expected['daily']]))))
if error>1e-7 or daily>1e-9:raise AssertionError('Endpoint reload failed')
(r/'endpoint_reload_audit.json').write_text(json.dumps({'status':'passed','step':c['step'],'probability_error':error,'daily_error':daily},indent=2))
