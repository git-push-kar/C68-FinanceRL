from datetime import datetime,timezone
from pathlib import Path
import json
R=Path(__file__).parent;rows=[]
for label,name in [('Nifty static native','static_native_seed_comparison_budget64.json'),('Nifty limited timing','regime_native_seed_comparison_budget64.json'),('Nifty warmup then RL (vs warmup)','warmup_native_seed_comparison_budget64.json'),('Five-asset static numerical','multiasset_static_comparison64.json'),('Five-asset dynamic numerical','multiasset_dynamic_comparison64.json'),('Five-asset static native','multiasset_native_comparison64.json'),('Five-asset dynamic native','multiasset_dynamic_native_comparison16.json'),('Direct asset weights, static CPU','asset_allocation_static_comparison64.json'),('Direct asset weights, dynamic CPU','asset_allocation_dynamic_comparison64.json'),('Worst-year reward, static CPU','asset_robust_static_comparison64.json'),('Worst-year reward, dynamic CPU','asset_robust_dynamic_comparison64.json'),('Causal risk anchor, static CPU','risk_anchor_static_comparison64.json'),('Causal risk anchor, dynamic CPU','risk_anchor_dynamic_comparison64.json'),('Nifty rotated labels native','label_rotate_comparison64.json'),('Nifty reversed labels native','label_reverse_comparison64.json'),('Five-asset readout-only native','readout_only_ablation_comparison16.json'),('Risk-cost reward, static CPU','asset_risk_reward_static_comparison64.json'),('Risk-cost reward, dynamic CPU','asset_risk_reward_dynamic_comparison64.json'),('Risk-tool selector, static CPU', 'risk_selector_static_comparison64.json'),('Risk-tool selector, dynamic CPU', 'risk_selector_dynamic_comparison64.json'),('Risk-tool selector, native', 'risk_selector_native_comparison64.json'),('Native full continued32', 'native_full_continued_comparison32.json'),('Native readout continued32', 'native_readout_continued_comparison32.json'),('Annual static CPU 2022', 'multiasset_annual_static_2022_comparison64.json'),('Annual static CPU 2023', 'multiasset_annual_static_2023_comparison64.json'),('Annual static CPU 2024', 'multiasset_annual_static_2024_comparison64.json'),('Annual dynamic CPU 2022', 'multiasset_annual_dynamic_2022_comparison64.json'),('Annual dynamic CPU 2023', 'multiasset_annual_dynamic_2023_comparison64.json'),('Annual dynamic CPU 2024', 'multiasset_annual_dynamic_2024_comparison64.json'),('Risk selector rotated-label control','risk_label_rotate_comparison64.json'),('Risk selector reversed-label control','risk_label_reverse_comparison64.json')]:
 p=R/'reports'/name
 if p.exists():
  d=json.loads(p.read_text());rows.append({'method':label,'mean_Sharpe_gain':d['mean_Sharpe_gain'],'positive_seeds':d['positive_seeds'],'seeds':d['training_seeds'],'period':d['dates'],'CI20':d['paired_95_intervals']['20']['shared_time_blocks_and_resampled_seeds'],'source':str(p)})
status={}
for name in ['multiasset_controls_status','multiasset_native_status','multiasset_dynamic_native_status','asset_allocation_controls_status','asset_robust_controls_status','risk_anchor_controls_status','label_robustness_status','readout_only_ablation_status','asset_risk_reward_controls_status','native_diagnostics_status','risk_selector_controls_status','multiasset_annual_controls_status','risk_selector_native_status','native_matched_extension_status','risk_label_robustness_status']:
 p=R/'reports'/f'{name}.json'
 if p.exists():status[name]=json.loads(p.read_text())
receipt_path=R/'reserved_finance_native_2026_access.json'
receipt=json.loads(receipt_path.read_text()) if receipt_path.exists() else None
final_result=None
if receipt:
 p=Path(receipt['contract']).parent/'result.json'
 if p.exists():
  result=json.loads(p.read_text())
  if result.get('seal_sha256')!=receipt['seal_sha256']:raise ValueError('Final receipt/result identity mismatch')
  final_result=result
data={'updated_utc':datetime.now(timezone.utc).isoformat(),'scope':'Development comparison rows only. Different datasets, dates, and initial policies; no cross-row absolute ranking. Development uncertainty is unadjusted for method search.','results':rows,'jobs':status,'reserved2026_access_started':receipt is not None,'reserved2026_evaluated':final_result is not None,'reserved2026_receipt':receipt,'reserved2026_result':final_result}
(R/'reports/FINAL_CAMPAIGN_STATUS.json').write_text(json.dumps(data,indent=2))
lines=['# Finance completion campaign status','',data['updated_utc'],'',data['scope'],'','| Method | Mean paired Sharpe gain | Positive seeds | 20-day paired interval |','|---|---:|---:|---|']
for r in rows:lines.append(f"| {r['method']} | {r['mean_Sharpe_gain']:+.5f} | {r['positive_seeds']}/{r['seeds']} | [{r['CI20'][0]:.5f}, {r['CI20'][1]:.5f}] |")
final_note=('No reserved2026 evaluation or final financial acceptance yet.' if not receipt else
 'Reserved2026 access has started; evaluation is incomplete. This period must not be reused for method selection.' if final_result is None else
 'Reserved2026 evaluation completed. See its sealed result; completion alone is not financial success. No retuning on these outcomes.')
lines+=['',final_note,'']
for k,v in status.items():lines.append(f"- {k}: {v.get('state', 'completed seeds ' + str(v.get('completed_seeds', [])))}; {v.get('job','no active job')}")
(R/'reports/FINAL_CAMPAIGN_STATUS.md').write_text('\n'.join(lines))
