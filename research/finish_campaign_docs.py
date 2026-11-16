"""Refresh navigation after the frozen campaign, without changing its protocol."""
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).parent

def main():
    folder = ROOT/'campaign_runs/reserved_post_publication_c4f2f205fb74'
    if json.loads((folder/'status.json').read_text())['status'] != 'completed':
        raise RuntimeError('Frozen evaluation incomplete')
    analysis = json.loads((ROOT/'reports/final_analysis.json').read_text())
    choice = json.loads((ROOT/'METHOD_CHOICE.json').read_text())
    seed = choice['deployment_seed']
    package = choice['strict_artifacts'][f"{choice['selected_family']}{seed}"]['package']
    header = f'''# Completed experiment; financial-quality goal not achieved

The adapter executes correctly, but final performance failed the user's
financial-quality requirement. Three-seed mean return -0.839%, Sharpe -0.0816,
drawdown 5.327%, versus equal-weight +9.335%, Sharpe 1.2408, drawdown 4.566%.
The GPU is not the explanation. See ASTRA_REVIEW_REQUEST.md for the next review.

Read **reports/FINAL_REPORT.md** first. It contains the frozen final results,
all three primary seeds, matched controls, uncertainty, failures and limits.
The final period is now USED: do not retune against it and call it unseen.

The deployment default is seed {seed}, selected by pre-2023 risk loss before
final outcomes. Complete package: `{package}`. Use covariance_trial/inference.py
with the pristine pinned InternVL3-2B and checks/router_smoke_deployment/request.json
as the integration example. LoRA alone is insufficient; numeric heads and
configuration are required. This outputs portfolio weights, not financial text.

Method: direct causal log relative portfolio variance, 49 fixed stocks,
252-return Ledoit-Wolf covariance, 30-day features, all 28 language layers,
rank-16 attention LoRA, 21-session rebalances, .30 target cap. Active math is
strict FP32 with both TF32 permissions false. No capacity or precision
reduction was made for this GPU. This is risk allocation, not PPO/GRPO.

Training ends in 2022. Reused 2023-2024 development is separate from the frozen
May-2025-January-2026 test. Final execution delays the complete signal one close,
uses actual current holdings and charges 10 bp times traded-stock L1.
METHOD_CHOICE.json and METHOD_CHOICE_STRICT_PRECISION.md are frozen records;
the final protocol is checks/FINAL_FROZEN_PROTOCOL_20261107/protocol.json.

The legacy cuDNN precision invariance audit FAILED and remains recorded.
The same primary method and matched no-LoRA controls were replicated from
initialization with both TF32 permissions disabled. All 75 runtime tests,
strict hardware gate, reload parity and real deployment-router checks passed.

Use .venv/Scripts/python.exe on this PC. Set FINANCERL_EXECUTION=selective_fp32,
FINANCERL_BASE_MODEL to local_models/InternVL3-2B, both offline flags to 1,
OMP_NUM_THREADS=2 and MKL_NUM_THREADS=2. Heavy compute pauses 22:50-07:00 IST;
no fan or persistent power-plan settings were changed. Do not launch historical
automatic experiment drivers. Runtime and selected weights are frozen.

Portable deliverable: deliverables/finance_research_bundle_20261107.zip.
Its external verification receipt is reports/PORTABLE_VERIFICATION.json when
present. The archive excludes the base and venv. Original verified base is
F:/pc/LY_C68/deepfake_project/models/InternVL3-2B. Recreate environments on
other machines and verify base identity; do not substitute a merged base.

backup_completed.py copies evidence and deliverables to
F:/Polyvalent_Continuation_2026-11-07, preserving the original handover.
STATUS.md is a timestamped snapshot. Exact attempt source, logs and exit codes
are under launch_records; all outcomes and selected controls are retained.

General finance QA, live trading and real four-domain integration remain
unfinished. Fixed-universe survivorship and incomplete original corporate-
adjustment provenance limit financial claims. Positive returns alone do not
establish an LLM contribution; use the matched controls and intervals.
'''
    old = ROOT/'CURRENT_HANDOFF.md'
    archive = ROOT/'reports/CURRENT_HANDOFF_BEFORE_FINAL.md'
    if old.exists() and not archive.exists(): shutil.copy2(old, archive)
    for name in ('START_HERE_CAMPAIGN.md','CURRENT_HANDOFF.md'):
        (ROOT/name).write_text(header, encoding='utf-8')
    p = ROOT/'DECISION_LOG.md'
    marker = '## Frozen final evaluation completed'
    text = p.read_text(encoding='utf-8')
    if marker not in text:
        text += '\n\n'+marker+'\n\n'+datetime.now(timezone.utc).isoformat()+'\n\n'
        text += ('All frozen entries completed. Models and runtime were not changed after final access. '
                 'Daily outcomes, paired temporal uncertainty and the full report are retained. '
                 'The deployment default remains the pre-final seed 123. No final-period winner is promoted.\n')
        p.write_text(text, encoding='utf-8')
    p = ROOT/'LITERATURE_COMPARISON.md'
    text = p.read_text(encoding='utf-8')
    if '## Completed strict final comparison' not in text:
        text += ('\n\n## Completed strict final comparison\n\nThe earlier tables above are development history. '
                 'The completed strict-precision primary and all frozen post-publication controls are in '
                 'reports/FINAL_REPORT.md and reports/final_analysis.json. Those same-data comparisons '
                 'decide the local claim; external paper headline Sharpe values cannot rank different markets. '
                 'FinATOM, Zhang et al. and the Nifty journal primary records were rechecked during final evaluation.\n')
        p.write_text(text, encoding='utf-8')
    print(json.dumps({'navigation':'updated','default_seed':seed,'package':package,
                      'final_mean':analysis['candidate_seed_mean']},indent=2))

if __name__ == '__main__': main()
