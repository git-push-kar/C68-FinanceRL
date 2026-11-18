from pathlib import Path
import json,shutil,hashlib,zipfile
R=Path.cwd();D=R.parent/'PolyvalentDemo';O=R/'deliverables/finance_final_figures_20261116'
text='''# Finalization and full-period figure addendum — 16 November 2026

## Decision

Freeze the selected original Finance RL adapter as the final artifact for this campaign, and report a mixed outcome. This is a decision to conclude the current experiment, not proof of optimality or consistent financial superiority. Do not describe 6.38% as a model-wide return ceiling or the highest-return run. It is the selected adapter's return on the original reserved 2026 interval. Explanation reliability remains a separate unresolved limitation; finalization does not mean every original quality objective was achieved.

## What data belongs in the results

RL weight training used 2010–2018. Validation 2019–2021 selected checkpoints. Development 2022–2024 was reused adaptively to design/select methods. Thus “not used for gradient training” does not mean untouched or independent. Show validation/development only with those labels. The main generalization result remains the original reserved 2 February–30 October 2026 test, chosen before evaluation. Report three trained seeds; deployment seed42 was fixed beforehand, not chosen from test scores.

Include all completed comparable frozen-adapter additional annual evaluations (2022, 2023, 2024, partial 2025) alongside that main result, including negative outcomes. The 2025 data had been accessed in other project campaigns and is an additional retrospective audit, not a new pristine test. Its available saved panel lacks sufficient warmup for January1 under the unchanged environment, so evaluation starts January10. Do not silently treat it as a full calendar year.

Annual portfolios reset to cash. They cannot be multiplied or stitched into one continuous investable result; such a result would require a separate continuous simulation with holdings carried across boundaries. Do not pool development and test into a headline score. Do not select another checkpoint or dates because they produce a prettier chart. No new evaluation of validation years or every available historical date is claimed by this figure package. All completed same-policy comparisons are included in its CSVs; historical experiments on other assets/methods remain in the main run history, not falsely merged with this adapter.

## Figures for PPT/reports

01_final_metrics: six panels for net return, zero-risk-free annualized Sharpe, maximum drawdown, annualized volatility, average daily full-L1 turnover and modeled dollar fees. Identical 189-session dates and execution for untrained, trained seed42, standalone minimum variance and equal weight. Mean turnover includes non-rebalance days. Fees are already deducted from returns. Volatility is population daily standard deviation times sqrt(252). The three-seed mean Sharpe and paired interval are in the caption; the bars use deployment seed42 consistently. Full metrics for all seeds are supplied separately.

02_final_equity_drawdown: actual saved daily portfolio values and drawdowns, beginning with $100,000 cash before the first return. No smoothing or synthetic path.

03_period_robustness: all five completed yearly/partial-year periods, separately labeled development, additional and reserved. Return is period total, not annualized. All annual comparisons use the same frozen seed42 policy against its exact untrained probabilities. Dates, period lengths, simple baselines and remaining metrics are in period_metrics.csv and the original annual audit.

Each figure is supplied as high-resolution PNG, vector SVG and PDF. CSVs contain the numbers; SOURCE_MANIFEST.json identifies input-file SHA256 hashes. make_final_finance_figures.py reproduces figures. No base or adapter weight was changed.

## Final interpretation and numbers

Period: untrained return → trained return; untrained Sharpe → trained Sharpe; untrained max drawdown → trained max drawdown.
- 2022 development: 4.585% → 4.965%; 0.583 → 0.807; 5.047% → 3.250%.
- 2023 development: 7.445% → 7.004%; 1.380 → 1.597; 3.739% → 2.982%.
- 2024 development: 12.225% → 12.057%; 2.325 → 2.810; 3.170% → 2.481%.
- Partial 2025 additional audit: 11.914% → 8.560%; 1.540 → 1.273; 6.258% → 5.792%.
- Original 2026 reserved test, seed42: 6.532% → 6.377%; 1.513 → 1.951; 3.509% → 2.490%.

Trained three-seed mean final Sharpe1.951463, raw1.513051. Paired95% ΔSharpe interval [-0.080094,0.864912] includes zero. Standalone minimum variance final Sharpe2.169516, return6.396%, drawdown2.195%; it outperforms the learned portfolio on all three of these metrics in this interval. The selected trained policy is not superior to this simple comparator.

Across the five reported periods, trained drawdown was lower in5/5; net return higher in1/5; Sharpe higher in4/5. These are descriptive counts across dependent, partly adaptively reused historical intervals, not independent statistical trials or a significance test. The 2025 weakness must be visible in the presentation.

## Final showable conclusion

A small InternVL3-2B RL adapter learned a more defensive mixture of causal allocation strategies. In the reserved evaluation it reduced volatility, drawdown, turnover and fees, improved observed Sharpe, and slightly reduced net return. Across additional historical regimes, the return and Sharpe benefits were inconsistent. Classical minimum variance remains a strong and sometimes better alternative. Generated portfolio explanations are a separate model capability with documented factual failures. These results support a transparent adaptation study, not a profit-improvement, universal superiority or reliable autonomous-adviser claim.
'''
(O/'README.md').write_text(text,encoding='utf-8');shutil.copy2(R/'make_final_finance_figures.py',O/'make_final_finance_figures.py')
V=R/'deliverables/finance_context_transfer_20261116_v4'
shutil.copytree(R/'deliverables/finance_context_transfer_20261116_v3',V)
shutil.copytree(O,V/'final_presentation_figures');shutil.copytree(R/'additional_frozen_adapter_audit_20261116',V/'additional_frozen_adapter_audit')
p=V/'CONTEXT_TRANSFER.md';p.write_text('# Current finalization update: read the closing addendum and final_presentation_figures/README.md. Earlier historical claims are superseded where inconsistent.\n\n'+p.read_text(encoding='utf-8')+'\n\n---\n\n'+text,encoding='utf-8')
(V/'TRANSFER_MANIFEST.json').write_text(json.dumps({p.relative_to(V).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in V.rglob('*') if p.is_file() and p.name!='TRANSFER_MANIFEST.json'},indent=2))
S=Path('F:/Polyvalent_Finance_Context_2026-11-16_v4');shutil.copytree(V,S)
with zipfile.ZipFile(str(S)+'.zip','x',compression=zipfile.ZIP_DEFLATED) as z:
 for p in S.rglob('*'):
  if p.is_file():z.write(p,p.relative_to(S).as_posix())
shutil.copytree(O,Path('F:/Finance_Final_Graphs_2026-11-16'))
status='\n\nCurrent finalization addendum and graphs: F:/Polyvalent_Finance_Context_2026-11-16_v4/CONTEXT_TRANSFER.md and F:/Finance_Final_Graphs_2026-11-16. Includes all completed 2022–2026 frozen-policy annual evaluations, with data-use labels and mixed outcomes. Original v3 archives are historical snapshots.\n'
for p in [D/'FINANCE_GENERATION_CURRENT_HANDOVER.md',R/'FINANCE_GENERATION_CURRENT_HANDOVER.md',Path('F:/FINANCE_GENERATION_CURRENT_HANDOVER.md'),R/'START_HERE_FINANCE_FINAL_20261116.md']:
 p.write_text(p.read_text(encoding='utf-8')+status,encoding='utf-8')
print('Saved v4 handover and SSD graph package')
