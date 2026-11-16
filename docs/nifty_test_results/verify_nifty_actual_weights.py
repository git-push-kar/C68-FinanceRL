import os,sys,json,gc,time
from pathlib import Path
import numpy as np
R=Path.cwd();O=R/'reports/profit_improving_runs_audit_20261116';sys.path.insert(0,str(R/'rl_candidate_static_native'))
os.environ.update(FINANCERL_EXECUTION='selective_fp32',FINANCERL_BASE_MODEL=str(R/'local_models/InternVL3-2B'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
import execution_policy
execution_policy.wait_for_compute_window=lambda:None
execution_policy.quiet_now=lambda moment=None:False
execution_policy.configure_finance_precision()
import torch
from shared_backbone import SharedBackbone
from text_policy import TextPolicy
from candidate_market import Market,tensor
from model import digest
torch.set_num_threads(2)
report=json.loads((R/'reports/static_native_seed_comparison_budget64.json').read_text());market=Market(str(R/'data/nifty50_prices.csv'),'2025-01-01','2025-12-31');obs=tensor(market.reset()[0],'cuda');started=time.time();shared=SharedBackbone.load('cuda','fp32');results=[]
for entry in report['runs']:
 run=Path(entry['run']);step=entry['selected_step']
 for k in [0,step]:
  package=run/f'checkpoint_{k}';policy=TextPolicy(market.env.tickers,package=package,shared=shared)
  with torch.inference_mode():
   actual=policy(obs)[0].probs[0].cpu().double().numpy();expected=np.array(json.loads((run/f'development_{k}.json').read_text())['probabilities'][0]);error=float(abs(actual-expected).max())
   with shared.model.disable_adapter():disabled=policy(obs)[0].probs[0].cpu().double().numpy()
  item={'run':run.name,'step':k,'probabilities':actual.tolist(),'saved_probabilities':expected.tolist(),'max_probability_error':error,'disabled_adapter_probabilities':disabled.tolist(),'adapter_sha256':digest(package/'finance_lora/finance/adapter_model.safetensors')};results.append(item);print(json.dumps(item),flush=True)
  shared.model=shared.model.unload();del policy;gc.collect();torch.cuda.empty_cache()
(O/'actual_weight_reload.json').write_text(json.dumps({'elapsed_seconds':time.time()-started,'execution':'Full pinned original InternVL3-2B; active decoder FP32, TF32 disabled; inactive full components retained on CPU; no quantization','entries':results},indent=2));print('DONE',flush=True)
