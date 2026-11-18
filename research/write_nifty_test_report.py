from pathlib import Path
import json,numpy as np,shutil,hashlib,zipfile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
R=Path.cwd();O=R/'reports/nifty_final_candidate_test_20261116';rows=json.loads((O/'metrics.json').read_text());G=O/'presentation';G.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.linewidth':.65,'axes.spines.top':False,'axes.spines.right':False,'xtick.major.size':0,'ytick.major.size':3,'svg.fonttype':'none','pdf.fonttype':42})
names=['untrained','seed42_64','buy_hold_equal'];colors=['#B6BCC5','#3569A8','#C58D42'];labels=['Untrained','Trained','Buy-and-hold']
fig,axes=plt.subplots(1,4,figsize=(7.6,2.8));fig.subplots_adjust(left=.06,right=.99,bottom=.16,top=.76,wspace=.6)
for i,(ax,key,factor,title) in enumerate(zip(axes,['return_total','sharpe_proxy','max_drawdown','volatility_daily'],[100,1,100,100*np.sqrt(252)],['Net return (%)','Sharpe ratio','Max. DD (%)','Ann. vol. (%)'])):
 vals=[next(x for x in rows if x['interval']=='test_2025' and x['fee_bps']==10 and x['policy']==n)[key]*factor for n in names]
 bars=ax.bar(range(3),vals,width=.62,color=colors,edgecolor='#333333',linewidth=.45,zorder=3);ax.set_ylim(0,max(vals)*1.22);ax.set_xlim(-.7,2.7);ax.yaxis.set_major_locator(MaxNLocator(nbins=4,min_n_ticks=3));ax.set_xticks([]);ax.set_title(f'({chr(97+i)}) {title}',fontsize=9,pad=10);ax.tick_params(axis='y',labelsize=8);ax.yaxis.grid(True,color='#E8E8E8',lw=.5,zorder=0)
 for b,v in zip(bars,vals):ax.text(b.get_x()+b.get_width()/2,v+max(vals)*.03,f'{v:.2f}',ha='center',fontsize=8)
fig.legend(handles=[Patch(facecolor=c,edgecolor='#333333',linewidth=.45,label=n) for c,n in zip(colors,labels)],loc='upper center',bbox_to_anchor=(.5,1),ncol=3,frameon=False,fontsize=9,handlelength=1.2)
for ext in ['png','svg','pdf']:fig.savefig(G/f'nifty_test_2025.{ext}',dpi=400,facecolor='white')
plt.close(fig)
caption='Figure. Retrospective test performance of the fixed Nifty49-stock InternVL3-2B strategy-preference adapter, seed42 checkpoint44, versus its original untrained probabilities and equal-weight buy-and-hold. Training2019–2021; validation/checkpoint selection2022; test1January–31December2025 (249 sessions). This candidate was revisited after historical experimentation and the test data had previously been accessed in other project campaigns; this is not untouched confirmation. Same vendor-adjusted stock panel, dates and10bp purchase/sale costs; no terminal liquidation, slippage, taxes or market impact. Equal-weight buy-and-hold makes one fee-funded initial purchase and holds fixed units. Sharpe assumes zero risk-free return. Annualized volatility is daily population standard deviation times sqrt252. All three trained seeds have essentially identical test point estimates; see report for uncertainty, higher-cost tests, January2026, and stronger rebalanced equal-weight/inverse-volatility controls.\n'
(G/'CAPTION.md').write_text(caption,encoding='utf-8')
text='''# Finance RL: verified candidate and training/validation/test results

## Recommendation

The Nifty native static strategy-preference adapter (seed42 checkpoint44) is the stronger candidate for demonstrating a training benefit. It improves test net return and Sharpe over the exact untrained policy, and slightly exceeds equal-weight buy-and-hold at the original10bp costs. Actual adapter weights were reloaded and verified. Do not claim it beats all simple trading strategies: rebalanced equal weight and inverse volatility remain stronger controls. No UI/deployment switch was made in this audit; the existing dashboard remains the old five-ETF experiment.

## Data partition for the final presentation

- Training:2019–2021, updates to LoRA parameters through financial RL.
- Validation:2022, checkpoint selection using net Sharpe. Selected steps42/123/456:44/40/44.
- Test:calendar2025 and January2026, shown separately. Also report a continuous1January2025–30January2026 test as supplementary material, carrying holdings through the year boundary.

2023–2024 development scores are EXCLUDED from the main results and figures. Their prior use for method design must remain disclosed in the methods/history. These test results are retrospective: the candidate was revisited after observing earlier experiments and the market data had been accessed elsewhere in the project. Changing terminology cannot turn them into independent untouched tests. Do not call validation or previously reused development history independent test evidence.

## Test2025:249 sessions,10bp fees

| Policy | Net return | Sharpe | Max drawdown |
|---|---:|---:|---:|
| Original untrained |13.274%|1.103|8.135%|
| Trained seed42 |15.074%|1.243|7.780%|
| Trained seed123 |15.074%|1.243|7.780%|
| Trained seed456 |15.074%|1.243|7.780%|
| Equal-weight buy-and-hold |14.864%|1.220|7.750%|
| Hold trained first target |14.741%|1.222|7.733%|
| Hold untrained first target |13.697%|1.136|8.120%|
| Equal-weight rebalanced |15.242%|1.244|7.787%|
| Inverse-volatility rebalanced |15.106%|1.252|7.754%|

The trained policy gains about1.80 percentage points versus untrained, and0.21 points versus equal-weight buy-and-hold. Its Sharpe gain over raw is about0.140. Its drawdown is slightly worse than buy-and-hold. All three trained seeds improve versus raw, but they share one market path and have near-identical strategies, so they are optimization replications, not independent market replications.

Conditional paired circular-block bootstrap,5000 resamples,20-session blocks: trained mean minus untrained return95% CI[0.590,3.264] percentage points; Sharpe CI[0.058,0.212]. Five/60-session sensitivity intervals also exclude zero for the raw comparison. These intervals do NOT adjust for historical method/candidate search and do not establish independent confirmatory significance. Versus equal-weight buy-and-hold, return CI[-0.880,1.277] points includes zero. Do not claim established superiority over holding.

## January2026 and continuous test

January2026 (21 sessions, independent cash reset): untrained-1.915%, trained-1.614%, buy-and-hold-1.588%. Training reduces the loss versus raw but does not beat holding. The month is too short for a strong standalone inference; bootstrap blocks comparable to or longer than this month are not credible independent confirmation.

Continuous1January2025–30January2026 (270 sessions, no year-end reset): untrained11.221%, trained13.283%, buy-and-hold13.059%, equal-weight rebalanced13.441%, inverse-volatility rebalanced13.343%. Report alongside the separate years; do not multiply reset-period returns and call them a continuous portfolio.

## Cost sensitivity

2025 test: at20bp, untrained12.563%, trained14.820%, buy-and-hold14.750%; at50bp, untrained10.458%, trained14.060%, buy-and-hold14.407%. The training gain over raw survives these modeled costs, but the small advantage over holding does not. These are fee sensitivity simulations, not a realistic market-impact model.

## What actually changed in the model

The same pinned full InternVL3-2B was loaded with FP32 active decoder and TF32 disabled; inactive original components remained on CPU. The six original packages (initial and selected for each seed) passed manifest verification. Maximum saved-versus-reloaded action-probability discrepancy was below1.5e-8. Disabling each trained adapter returned the same original untrained probabilities. This verifies real weight adaptation, rather than a manually substituted allocation.

For seed42, strategy probabilities changed approximately:
- Equal weight20.57% →32.88%.
- Inverse volatility31.39% →64.49%.
- Momentum30.48% →1.36%.
- Reversal17.56% →1.28%.

These probabilities are fixed strategy preferences; causal strategy target weights change with prices and the portfolio rebalances every five sessions. Training learned a more useful strategy mixture on these test histories. It did not learn daily news reasoning or prove state-dependent timing. The observed raw-relative gain includes lower turnover/costs, not exclusively superior gross stock selection.

## Verification and boundaries

Source stock-data SHA256 exactly matches the original training protocol. The original fixed49-stock universe and preprocessing remain; current-universe/survivorship and point-in-time limitations are not solved. All11 policies in each of nine period/cost combinations share exact return dates. Daily-return compounding and ending wealth reconcile; all hold baselines have zero turnover after the initial purchase. No training or original artifact modification occurred. Actual inference probabilities, full daily paths, bootstrap intervals, protocols and accounting checks accompany this report.

Main figures: presentation/nifty_test_2025.png, SVG and PDF; caption separate. Main presentation contains training/validation/test definitions and test results only. Historical development figures remain in prior archives for provenance.

## Defensible final statement

Financial RL changed the native2B model's strategy preferences and improved subsequent test return and Sharpe versus its original untrained policy across three training seeds. It slightly outperformed equal-weight buy-and-hold at the original fee setting over2025 and the continuous available test interval. That holding advantage is small, uncertain and cost-sensitive, and simple rebalanced strategies still match or outperform it. This supports an adaptation-benefit claim, not universal portfolio-management superiority.
'''
(O/'FINAL_TEST_REPORT.md').write_text(text,encoding='utf-8')
shutil.copy2(R/'reports/profit_improving_runs_audit_20261116/actual_weight_reload.json',O/'actual_weight_reload.json')
for name in ['verify_nifty_actual_weights.py','test_nifty_candidate.py']:shutil.copy2(R/name,O/name)
(O/'MANIFEST.json').write_text(json.dumps({p.relative_to(O).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in O.rglob('*') if p.is_file() and p.name!='MANIFEST.json'},indent=2))
V=R/'deliverables/finance_context_transfer_20261116_v5';shutil.copytree(R/'deliverables/finance_context_transfer_20261116_v4',V)
shutil.copytree(O,V/'nifty_test_results')
old=(V/'CONTEXT_TRANSFER.md').read_text(encoding='utf-8');(V/'CONTEXT_TRANSFER.md').write_text(text+'\n\n---\n\n# Historical research context: superseded primary ETF selection retained for provenance\n\n'+old+'\n\n---\n\n# Current presentation result\n\nUse nifty_test_results/FINAL_TEST_REPORT.md and presentation/nifty_test_2025.*. Exclude development scores from the main presentation. The verified Nifty adapter shows training benefit versus raw on retrospective tests; its small buy-and-hold advantage is not statistically established. The running UI still uses the earlier ETF adapter; do not present its records as this Nifty result.\n',encoding='utf-8')
(V/'TRANSFER_MANIFEST.json').write_text(json.dumps({p.relative_to(V).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in V.rglob('*') if p.is_file() and p.name!='TRANSFER_MANIFEST.json'},indent=2))
S=Path('F:/Polyvalent_Finance_Context_2026-11-16_v5');shutil.copytree(V,S)
with zipfile.ZipFile(str(S)+'.zip','x',compression=zipfile.ZIP_DEFLATED) as z:
 for p in S.rglob('*'):
  if p.is_file():z.write(p,p.relative_to(S).as_posix())
shutil.copytree(O,Path('F:/Finance_Nifty_Test_Results_2026-11-16'))
status='''# Current Finance research result — v5

Read F:/Finance_Nifty_Test_Results_2026-11-16/FINAL_TEST_REPORT.md and F:/Polyvalent_Finance_Context_2026-11-16_v5/CONTEXT_TRANSFER.md. Stronger verified candidate: Nifty native static seed42 checkpoint44, with seeds123/456 replications. Training2019–2021; validation2022; retrospective tests2025 and January2026. No development scores in the main result. All actual packages reloaded; disable-adapter recovers raw. 2025 return raw13.274%, trained15.074%, hold14.864%; continuous available test raw11.221%, trained13.283%, hold13.059%. Small, uncertain buy-and-hold advantage; simple rebalanced strategies stronger. Full report includes cost sensitivity and prior-test-access caveat. UI still displays older ETF result; Nifty deployment integration not done. No training running. Older headers/docs/archives are historical.\n'''
for p in [R/'FINANCE_GENERATION_CURRENT_HANDOVER.md',R.parent/'PolyvalentDemo/FINANCE_GENERATION_CURRENT_HANDOVER.md',Path('F:/FINANCE_GENERATION_CURRENT_HANDOVER.md')]:p.write_text(status,encoding='utf-8')
print('Saved verified test report, scientific figure and v5 handover')
