"""Load the actual selected adapter; return weights for a verified controller request."""
import argparse
import json
import os
from pathlib import Path
import sys

p = argparse.ArgumentParser()
p.add_argument('--base-model', required=True)
p.add_argument('--request', default=str(Path(__file__).with_name('example_request.json')))
p.add_argument('--seed', type=int, choices=(42,123,456), default=42)
a = p.parse_args()
root = Path(__file__).resolve().parent
os.environ.update(FINANCERL_EXECUTION='selective_fp32', FINANCERL_BASE_MODEL=str(Path(a.base_model).resolve()),
                  HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
sys.path.insert(0, str(root/'runtime_source'))
import execution_policy
execution_policy.wait_for_compute_window = lambda: None
execution_policy.quiet_now = lambda moment=None: False
from text_policy import TextPolicy
from shared_backbone import SharedBackbone, DomainRouter
from native_finance_runtime import NativeFinanceSpecialist
import torch
torch.set_num_threads(2)
shared = SharedBackbone.load('cuda','fp32')
specialist = NativeFinanceSpecialist(root/'adapters'/f'seed{a.seed}', TextPolicy, shared)
router = DomainRouter(shared)
specialist.register(router)
print(json.dumps(router.predict('finance', json.loads(Path(a.request).read_text())), indent=2))
