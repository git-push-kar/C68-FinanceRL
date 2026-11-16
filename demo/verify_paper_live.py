import requests,json,time
from pathlib import Path
s=requests.Session();base='http://127.0.0.1:8765';r=s.post(base+'/finance/paper/sessions',json={'capital':100000});r.raise_for_status();j=r.json();key=j['session_id'];started=time.time()
while not j['finished']:
 r=s.post(base+f'/finance/paper/sessions/{key}/advance',json={'mode':'rebalance','expected_sessions':j['sessions']},timeout=180);r.raise_for_status();j=r.json()
p=Path('finance/evaluation/paper_full_live_run.json');p.write_text(json.dumps(j,indent=2));ref=json.loads(Path('finance/evaluation/paper_golden.json').read_text());err=max(abs(h['equity']-v) for h,v in zip(j['history'],ref['trained_equity']));assert err<.02
assert len(j['activities'])==54 and all(a['model_receipt']['mode']=='real_model' for a in j['activities']);print(json.dumps({'elapsed_seconds':time.time()-started,'days':j['sessions'],'actual_model_calls':len(j['activities']),'max_equity_difference_from_original':err,'final_equity':j['equity'],'return':j['net_return'],'fees':j['fees']},indent=2))
Path('finance/evaluation/paper_actual_run_receipt.json').write_text(json.dumps({'days':j['sessions'],'actual_model_calls':len(j['activities']),'max_equity_difference_from_original':err,'final_equity':j['equity']},indent=2))
