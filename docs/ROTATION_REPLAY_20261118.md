# Replaying a final rotation checkpoint

The portable archive contains three siblings named `repository`, `PolyvalentFinance` and `demo_adapters`, plus the historical checkpoint release. Copy the archive to local storage before starting integration. Run the following from its `repository` directory with the existing finance GPU Python environment. On another PC install the matching dependencies first. The market panel and execution fields resolve relative to the repository source.

```python
import json
from pathlib import Path
import numpy as np
from research.final_rotation import INSTRUCTION, NativePolicy, RotationMarket, load, simulate

assets = Path.cwd().parent / "PolyvalentFinance"
run = assets / "runs" / "final_rotation49_20261118"
policy = NativePolicy(assets / "local_models" / "InternVL3-2B")
policy.action_instruction = INSTRUCTION
market = RotationMarket()
for seed in (42, 123, 456):
    folder = run / f"seed{seed}"
    selected = json.loads((folder / "selection.json").read_text())
    load(policy, folder / selected["checkpoint"])
    replay = simulate(market, policy, "2025-01-01", "2026-01-30", 100000)
    saved = json.loads((folder / "test_100000_rl.json").read_text())
    np.testing.assert_allclose(replay["daily_equity"], saved["daily_equity"], rtol=0, atol=1e-7)
    assert replay["decisions"] == saved["decisions"]
    print(seed, selected["checkpoint"], replay["metrics"]["return"])
```

The loader verifies the base-model digest and the saved checkpoint manifest. Keep the numerical bridge, policy metadata and finance LoRA together. The existing [paper-trading simulation](../demo/PAPER_TRADING.md) uses its historical strategy-mixture adapter. The rotation research policy has a different observation, action and execution contract.
