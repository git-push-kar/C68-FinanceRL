# Constant-context native LoRA, 14 November 2026

Hypothesis: limiting timing flexibility avoids the overfit seen in the dynamic
summary control. This is stable strategy-mixture learning, not forecasting.
Original InternVL3-2B token probabilities A/B/C/D define the mixture. Only rank16
attention and original vocabulary-head LoRA train; no separate trainable head.
Use the unchanged constant prompt and original untrained probabilities, even if
they already favor a strong strategy. Never reset the native base to uniform.

Train2019-2021, validate2022, reused development2023-2024. Three seeds42/123/456,
64 full-history updates, assess every4, select highest validation net Sharpe,
including zero, earliest tie. No 2026 access. Full exact-fee portfolio Sharpe
with .01 KL to original policy; attention LR1e-5, vocabulary LoRA3e-4, cap1,
post-update KL ceiling .03. One full financial path per step. Constant context
allows exact aggregation of all decision gradients into one model backward.
No model cuts or approximate minibatch objective are required.

Technical gate requires frozen-base hashes, actual LoRA changes, exact initial
and reload behavior. All seeds retained regardless outcomes. CPU static control,
four fixed strategies, and simple mixture optimizer must remain visible; this
experiment cannot establish that an LLM is needed. Improvement on reused
development is preliminary, especially after adaptive method search.
