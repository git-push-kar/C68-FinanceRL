import argparse,json
from pathlib import Path
import numpy as np
import torch
from text_policy import TextPolicy
from candidate_market import Market,tensor,evaluate
from model import write_json
from execution_policy import configure_finance_precision,wait_for_compute_window
def main():
 a=argparse.ArgumentParser();a.add_argument('--run',required=True);r=Path(a.parse_args().run)
 wait_for_compute_window();configure_finance_precision();torch.set_num_threads(2)
 status=json.loads((r/'status.json').read_text());s=json.loads((r/'selection.json').read_text())['selected']
 if status['status']!='completed' or status['initial_lora_sha256']==status['final_lora_sha256']:raise AssertionError('No learning')
 m=Market('../data/nifty50_prices.csv','2023-01-01','2024-12-31');obs=tensor(m.reset()[0],'cuda');p=TextPolicy(m.env.tickers,s['package'])
 with torch.no_grad():trained=p(obs)[0].probs[0].cpu().numpy().astype(float)
 with torch.no_grad(),p.shared.model.disable_adapter():base=p(obs)[0].probs[0].cpu().numpy().astype(float)
 trained/=trained.sum();base/=base.sum()
 errors={}
 for key,step,probs in [('reload',s['step'],trained),('disabled',0,base)]:
  actual=evaluate(None,m,'cpu',fixed=probs);expected=json.loads((r/f'development_{step}.json').read_text())
  error=float(np.max(abs(np.asarray(actual['probabilities'])-np.asarray(expected['probabilities']))))
  daily=float(np.max(abs(np.asarray([x['net_return'] for x in actual['daily']])-np.asarray([x['net_return'] for x in expected['daily']]))))
  if error>1e-7 or daily>1e-9:raise AssertionError('Reload/base differs')
  errors[key]={'probability_error':error,'daily_error':daily}
 write_json(r/'reload_audit.json',{'status':'passed','checks':errors,'selected_step':s['step'],'scope':'Technical audit; no final financial acceptance'})
if __name__=='__main__':main()
