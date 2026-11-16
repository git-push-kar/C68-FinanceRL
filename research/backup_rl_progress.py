"""Preserve current RL source/data and completed evidence on the SSD."""
from datetime import datetime,timezone
from pathlib import Path
import json,shutil
root=Path(__file__).parent;target=Path('F:/Polyvalent_Continuation_2026-11-07')
if not (target/'CAMPAIGN_BACKUP.json').exists():raise RuntimeError('Unrecognized backup root')
for name in ['rl_strategy_trial','rl_sector_trial','rl_text_trial','rl_counterfactual_trial','rl_readout_trial','rl_numeric_control','rl_summary_numeric_control','rl_exposure_design','rl_exposure_trial','rl_exposure_soft_trial','rl_exposure_native_trial','rl_exposure_direct_trial','rl_exposure_direct_native_trial','rl_exposure_global_trial','rl_exposure_walkforward_trial','rl_candidate_trial','rl_candidate_static_trial','rl_candidate_static_native','rl_candidate_regime_native','rl_static_chronology','rl_static_warmup','rl_multiasset_static','rl_multiasset_dynamic','rl_multiasset_native','rl_multiasset_dynamic_native','rl_multiasset_readout_only','rl_multiasset_dynamic_resume','rl_multiasset_readout_resume','rl_multiasset_annual_static','rl_multiasset_annual_dynamic','rl_asset_risk_reward_static','rl_asset_risk_reward_dynamic','rl_asset_allocation_static','rl_asset_allocation_dynamic','rl_asset_robust_static','rl_asset_robust_dynamic','rl_risk_selector_static','rl_risk_selector_dynamic','rl_risk_selector_native','rl_risk_labels_rotate','rl_risk_labels_reverse','rl_risk_anchor_static','rl_risk_anchor_dynamic','rl_static_labels_rotate','rl_static_labels_reverse','multiasset_data_20261115','data_extension_20261107','sector_data_20261107']:
    shutil.copytree(root/name,target/name,dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__'))
for name in ['RL_CONTINUATION.md','FINANCE_RL_SUCCESS_CRITERIA.md','FINANCE_FINAL_CAMPAIGN.md','SMALL_MODEL_FINANCE_RL_REVIEW.md',
             'SECTOR_DATA_RATIONALE.md','CURRENT_HANDOFF.md','START_HERE_CAMPAIGN.md',
             'RL_FINAL_EVALUATION_DRAFT.md','RL_CONTINGENCY_HYPOTHESIS.md','NATIVE_FOLLOWON_DECISION.md',
             'READOUT_BUDGET_EXTENSION_20261108.md','ANNUAL_REFIT_DIAGNOSTIC_PENDING.md','NATIVE_BUDGET_EXTENSION_PENDING.md','NATIVE_RUNTIME_PENDING.md','RISK_SELECTOR_EXPERIMENT.md']:
    shutil.copy2(root/name,target/name)
shutil.copy2(root/'NATIVE_FINAL_EVALUATOR_PENDING.md',target/'NATIVE_FINAL_EVALUATOR_PENDING.md')
for name in ['native_static_evaluation_rehearsal','native_static_evaluation_rehearsal_v2']:
    path=root/'checks'/name
    if (path/'result.json').exists():shutil.copytree(path,target/'checks'/name,dirs_exist_ok=True)
if (root/'checks/native_static_evaluation_rehearsal_spec.json').exists():
    shutil.copy2(root/'checks/native_static_evaluation_rehearsal_spec.json',target/'checks/native_static_evaluation_rehearsal_spec.json')
for path in root.glob('*.py'):shutil.copy2(path,target/path.name)
for path in root.glob('*.ps1'):shutil.copy2(path,target/path.name)
for name in ['RL_PROGRESS_20261107_EVENING.md','RL_EVIDENCE_INDEX.md','rl_development_data_diagnostic.json',
             'sector_strategy_controls.json','numeric_ppo_seed_comparison.json','numeric_ppo_seed_comparison_budget128.json',
             'numeric_control_calibration.json','numeric_control_calibrated_seed_comparison.json',
             'internvl_ppo_seed_comparison_budget128.json',
             'nifty_validation_static_grid.json','sector_validation_static_grid.json',
             'matched_full_frozen_lora_initialization.json',
             'matched_ppo_replications.png','matched_ppo_replications.svg',
             'native_attention_all_action_seed_comparison.json','native_readout_all_action_seed_comparison.json',
             'native_followon_v1_source.txt','native_followon_v1_status.json','RL_PROGRESS_20261108.md',
             'native_readout_all_action_seed_comparison_budget256.json','summary_numeric_calibration.json',
             'summary_numeric_seed_comparison_budget256.json','summary_numeric_seed_comparison_budget1024.json',
             'strategy_overlap_training.json','exposure_declared_controls.json','exposure_numeric_seed_comparison_budget1024.json',
             'exposure_soft_numeric_seed_comparison_budget1024.json','exposure_training_signal.json','exposure_direct_numeric_seed_comparison_budget256.json','EXPOSURE_CPU_COMPARISON.md','RL_FAILURE_ANALYSIS_20261108.md','exposure_global_numeric_seed_comparison_budget64.json',
             'exposure_global_numeric_seed_comparison_budget256.json','global_queue_driver_executed.py',
             'exposure_walkforward_f1_seed_comparison_budget64.json','exposure_walkforward_f2_seed_comparison_budget64.json','exposure_walkforward_f3_seed_comparison_budget64.json','EXPOSURE_WALKFORWARD.md','exposure_walkforward_summary.json','candidate_numeric_seed_comparison_budget64.json','candidate_static_seed_comparison_budget64.json','static_native_seed_comparison_budget64.json','regime_native_seed_comparison_budget64.json','regime_vs_static_anchor.json','static_chronology_f1.json','static_chronology_f2.json','static_chronology_summary.json','STATIC_CHRONOLOGY.md','warmup_stage_comparison.json','static_cost_sensitivity.json','multiasset_static_comparison64.json','multiasset_dynamic_comparison64.json','multiasset_native_comparison64.json','multiasset_dynamic_native_comparison16.json','asset_allocation_static_comparison64.json','asset_allocation_dynamic_comparison64.json','asset_robust_static_comparison64.json','asset_robust_dynamic_comparison64.json','risk_anchor_static_comparison64.json','risk_anchor_dynamic_comparison64.json','asset_allocation_risk_controls.json','label_rotate_comparison64.json','label_reverse_comparison64.json','readout_only_ablation_comparison16.json','development_candidate_integrity.json','static_native_year_slices.json','multiasset_asset_regime_diagnostic.json','native_development_candidates.png','native_development_candidates.svg','asset_risk_reward_static_comparison64.json','asset_risk_reward_dynamic_comparison64.json','risk_selector_static_comparison64.json','risk_selector_dynamic_comparison64.json','risk_selector_native_comparison64.json','native_full_continued_comparison32.json','native_readout_continued_comparison32.json','multiasset_annual_static_2022_comparison64.json','multiasset_annual_static_2023_comparison64.json','multiasset_annual_static_2024_comparison64.json','multiasset_annual_dynamic_2022_comparison64.json','multiasset_annual_dynamic_2023_comparison64.json','multiasset_annual_dynamic_2024_comparison64.json','risk_label_rotate_comparison64.json','risk_label_reverse_comparison64.json','risk_selector_screen.json','multiasset_annual_summary.json','native_extension_trigger.json','system_clock_observation_20261115.json','multiasset_static_cost_sensitivity.json','native_fixed_cash_benchmark_sensitivity.json','FINAL_CAMPAIGN_STATUS.json','FINAL_CAMPAIGN_STATUS.md','warmup_native_seed_comparison_budget64.json','full_frozen_ppo_paired128.json','matched_full_frozen_lora_initialization_seed123.json','matched_full_frozen_lora_initialization_seed456.json']:
    path=root/'reports'/name
    if path.exists():
        (target/'reports').mkdir(exist_ok=True);shutil.copy2(path,target/'reports'/name)
for name in ['native_text_input_audit.json','readout_parameter_scope.json','native_readout_initialization_parity.json','exposure_actor_loss_initialization_parity.json','exposure_native_prompt_audit.json','exposure_direct_initialization_parity.json','exposure_direct_native_prompt_audit.json','exposure_global_initialization_parity.json','exposure_walkforward_prefix_parity.json']:
    path=root/'checks'/name
    if path.exists():
        (target/'checks').mkdir(exist_ok=True);shutil.copy2(path,target/'checks'/name)
for gate_name in ['native_text_rl_gate','readout_lora_rl_gate','candidate_static_native_gate_retry1','multiasset_native_gate','multiasset_dynamic_native_gate','multiasset_readout_only_gate','risk_selector_native_gate']:
    gate=root/'checks'/gate_name
    if (gate/'result.json').exists():shutil.copytree(gate,target/'checks'/gate_name,dirs_exist_ok=True)
for smoke_name in ['summary_numeric_smoke','exposure_numeric_smoke','exposure_soft_numeric_smoke','exposure_direct_numeric_smoke','exposure_global_numeric_smoke','exposure_walkforward_smoke','candidate_summary_smoke','multiasset_dynamic_native_smoke','multiasset_readout_only_smoke','asset_robust_static_smoke','asset_robust_dynamic_smoke','risk_anchor_static_smoke','risk_anchor_dynamic_smoke','asset_risk_reward_static_smoke','asset_risk_reward_dynamic_smoke','risk_selector_static_smoke','risk_selector_dynamic_smoke','multiasset_annual_static_parity','multiasset_annual_dynamic_parity']:
    smoke=root/'checks'/smoke_name
    if (smoke/'reload_audit.json').exists():shutil.copytree(smoke,target/'checks'/smoke_name,dirs_exist_ok=True)
copied=[];failed=[]
for folder in (root/'campaign_runs').iterdir():
    if not folder.name.startswith(('strategy_ppo','sector_','nifty_candidate_','nifty_static_','nifty_regime_','nifty_warmup_','multiasset_','asset_allocation_','asset_robust_','asset_risk_reward_','risk_anchor_','risk_selector_','risk_labels_','nifty_labels_')):continue
    status=folder/'status.json'
    launch=root/'launch_records'/folder.name/'status.json'
    try:state=json.loads((status if status.exists() else launch).read_text()).get('status')
    except (FileNotFoundError,json.JSONDecodeError):continue
    if state in ('completed','failed'):
        shutil.copytree(folder,target/'campaign_runs'/folder.name,dirs_exist_ok=True)
        (copied if state=='completed' else failed).append(folder.name)
for folder in (root/'launch_records').iterdir():
    if not folder.name.startswith(('strategy_ppo','sector_','native_text','readout_lora','audit_','endpoint_','nifty_candidate_','nifty_static_','nifty_regime_','nifty_warmup_','multiasset_','asset_allocation_','asset_robust_','asset_risk_reward_','risk_anchor_','risk_selector_','risk_labels_','nifty_labels_','candidate_','controls_','attribution_')):continue
    try:state=json.loads((folder/'status.json').read_text())
    except (FileNotFoundError,json.JSONDecodeError):continue
    if state['status'] in ('completed','failed'):
        shutil.copytree(folder,target/'launch_records'/folder.name,dirs_exist_ok=True)
result={'copied_utc':datetime.now(timezone.utc).isoformat(),'completed_rl_runs':copied,'failed_rl_runs_preserved':failed,
        'active_runs':'Not copied as completed; live source and completed records retained separately'}
(target/'RL_BACKUP_STATUS.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
