"""Fresh-process actual endpoint audit, even when selection chooses zero."""
import argparse,sys,json,os
from pathlib import Path
R=Path(__file__).parent
a=argparse.ArgumentParser();a.add_argument('--run',required=True);args=a.parse_args();run=Path(args.run).resolve()
sys.path.insert(0,str(R/'rl_candidate_regime_native'));os.chdir(R/'rl_candidate_regime_native')
import numpy as np
import torch
from text_policy import TextPolicy
from candidate_market import Market,tensor
from exposure_prompt import regime
from execution_policy import configure_finance_precision,wait_for_compute_window
wait_for_compute_window();configure_finance_precision();torch.set_num_threads(2)
cs=json.loads((run/'candidates.json').read_text());last=max(cs,key=lambda x:x['step'])
m=Market('../data/nifty50_prices.csv','2023-01-01','2024-12-31');p=TextPolicy(m.env.tickers,last['package']);obs=tensor(m.reset()[0],'cuda');probs=[]
for vol in (10.,20.,30.):
 obs['summary'][0,2]=vol
 with torch.no_grad():v=p(obs)[0].probs[0].double();v=v/v.sum();probs.append(v.cpu().numpy())
state=m.reset();history=[]
while True:
 ob,t=state;v=probs[regime(tensor(ob,'cpu'))];history.append(v);state,_,done=m.step(v@t)
 if done:break
expected=json.loads((run/f'development_{last["step"]}.json').read_text())
error=float(np.max(abs(np.asarray(history)-np.asarray(expected['probabilities']))))
daily=float(np.max(abs(np.asarray([i['net_return'] for i in m.infos])-np.asarray([i['net_return'] for i in expected['daily']]))))
if error>1e-7 or daily>1e-9:raise AssertionError('Actual trained endpoint differs')
(run/'endpoint_reload_audit.json').write_text(json.dumps({'status':'passed','step':last['step'],'probability_error':error,'daily_error':daily},indent=2))
