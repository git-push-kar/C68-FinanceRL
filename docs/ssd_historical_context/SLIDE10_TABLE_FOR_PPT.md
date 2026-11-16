# SLIDE 10 — PUT THIS DIRECTLY IN PPT

| Stage | Best Val Accuracy | Real Test (Unseen) |
|-------|-------------------|---------------------|
| **1. Scratch ViT** | 0.53 | ~0.50 (random) |
| **2. CLIP + FFT** | 0.90 | **0.49** (collapsed — predicts all fake) |
| **3. InternVL + LoRA** | 0.86 | **0.82** (CM 0.97 on unseen models) |

```
Stage 1: ████████░░░░░░░░  0.53
Stage 2: ████████████████░░  0.90 (training) → 0.49 (real test)  ← collapsed
Stage 3: ██████████████░░░░  0.86 (training) → 0.82 (real test)  ← generalization
```

Copy the table + bars as-is. That's your growth evidence: training jumped 0.53→0.90, but real test only jumped 0.49→0.82 after InternVL — proves the foundation fixed the collapse.
