"""Descriptive retrospective check of each seed's validation-best RL-only candidate."""
import json
from pathlib import Path
import torch
from research.counterfactual_rl.train import BASE,load,write
from research.counterfactual_rl.market import Market,evaluate
from research.counterfactual_rl.policy import NativePolicy

def main():
    root=Path.home()/'Documents/PolyvalentFinance/runs/counterfactual49_20261118'
    m=Market()
    p=NativePolicy(BASE)
    rows=[]
    for seed in (42,123,456):
        d=root/f'seed{seed}'
        candidates=json.loads((d/'candidates.json').read_text())
        best=max((c for c in candidates if c['stage']=='rl'),key=lambda c:c['score'])
        load(p,d/best['checkpoint'])
        result=evaluate(m,p.choose,'2025-01-01','2026-01-30',100000)
        rows.append(dict(seed=seed,checkpoint=best['checkpoint'],**result['metrics']))
    write(root/'secondary_rl_only.json',rows)

if __name__=='__main__':
    torch.set_num_threads(2)
    main()
