# Native InternVL Finance RL with numerical input adaptation

Conditional fallback only. Read
[the complete rationale, parameter changes and launch safeguards](../../docs/STATEFUL_NUMERIC_BRIDGE_CONTINUATION_20261117.md).

This model requires both the PEFT finance_lora directory AND numeric_bridge.safetensors.
The additional encoder feeds original decoder input embeddings; it does not emit
portfolio actions. Dropping this file produces a different policy. GPU validation
must pass before training. Conditional learning must pass before financial RL.
