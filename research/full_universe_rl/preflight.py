"""All49 native-model mechanical check; artificial quotes, no profit claims."""
import json,time
from pathlib import Path
import numpy as np
import torch
from state import OriginalPanel
from policy import NativePolicy

torch.set_num_threads(2);torch.manual_seed(42);start=time.perf_counter()
p=OriginalPanel();t=int(np.searchsorted(p.dates,np.datetime64('2024-01-02')))
o=p.observe(t,100000.,np.zeros(49,dtype=np.int64),100000.,np.linspace(100,5000,49))
model=NativePolicy('C:/Users/Parth/Documents/PolyvalentFinance/local_models/InternVL3-2B');e=model.encode(o)
with torch.no_grad():
 before=model(e).clone()
 with model.model.disable_adapter():raw=model(e).clone()
assert before.shape==(49,7) and float(abs(before-raw).max())==0
opt=torch.optim.AdamW([v for v in model.parameters() if v.requires_grad],lr=1e-4)
target=torch.arange(49,device='cuda')%7
prob=model(e);loss=-prob.gather(1,target[:,None]).log().mean();loss.backward()
bridge_grad=float(model.numeric_bridge[-1].weight.grad.abs().sum());assert bridge_grad>0
assert all(v.grad is None for v in model.parameters() if not v.requires_grad)
torch.nn.utils.clip_grad_norm_([v for v in model.parameters() if v.requires_grad],1.);opt.step();opt.zero_grad()
with torch.no_grad():
 after=model(e).clone()
 with model.model.disable_adapter():recovered=model(e).clone()
assert float(abs(before-after).max())>0 and float(abs(raw-recovered).max())==0
report={'status':'passed','scope':'Mechanical49-stock input/update test only. Artificial quotes and artificial action targets; not a learned financial policy or profitable result.','stocks':49,'numeric_input_shape':list(e['numeric_state'].shape),'output_shape':list(after.shape),'tokens':e['ids'].shape[1],'bridge_gradient_abs_sum':bridge_grad,'probability_change':float(abs(before-after).max()),'disabled_recovery_error':float(abs(raw-recovered).max()),'peak_gpu_bytes':torch.cuda.max_memory_allocated(),'elapsed_seconds':time.perf_counter()-start}
(Path(__file__).parent/'preflight_receipt.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)
