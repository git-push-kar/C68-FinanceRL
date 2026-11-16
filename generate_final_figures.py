from pathlib import Path
import json,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
G=Path.cwd();O=G/'docs/figures';O.mkdir(exist_ok=True);rows=json.loads((G/'docs/nifty_test_results/metrics.json').read_text());names=['untrained','seed42_64','buy_hold_equal'];colors=['#B6BCC5','#3569A8','#C58D42'];labels=['Untrained','Trained','Buy-and-hold']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.linewidth':.65,'axes.spines.top':False,'axes.spines.right':False,'xtick.major.size':0,'ytick.major.size':3,'svg.fonttype':'none','pdf.fonttype':42})
fig,axes=plt.subplots(1,4,figsize=(7.6,2.8));fig.subplots_adjust(left=.06,right=.99,bottom=.16,top=.76,wspace=.6)
for i,(ax,key,factor,title) in enumerate(zip(axes,['return_total','sharpe_proxy','max_drawdown','volatility_daily'],[100,1,100,100*np.sqrt(252)],['Net return (%)','Sharpe ratio','Max. DD (%)','Ann. vol. (%)'])):
 vals=[next(x for x in rows if x['interval']=='test_combined' and x['fee_bps']==10 and x['policy']==n)[key]*factor for n in names];bars=ax.bar(range(3),vals,width=.62,color=colors,edgecolor='#333333',linewidth=.45,zorder=3);ax.set_ylim(0,max(vals)*1.22);ax.set_xlim(-.7,2.7);ax.yaxis.set_major_locator(MaxNLocator(nbins=4,min_n_ticks=3));ax.set_xticks([]);ax.set_title(f'({chr(97+i)}) {title}',fontsize=9,pad=10);ax.tick_params(axis='y',labelsize=8);ax.yaxis.grid(True,color='#E8E8E8',lw=.5,zorder=0)
 for b,v in zip(bars,vals):ax.text(b.get_x()+b.get_width()/2,v+max(vals)*.03,f'{v:.2f}',ha='center',fontsize=8)
fig.legend(handles=[Patch(facecolor=c,edgecolor='#333333',linewidth=.45,label=n) for c,n in zip(colors,labels)],loc='upper center',bbox_to_anchor=(.5,1),ncol=3,frameon=False,fontsize=9,handlelength=1.2)
for ext in ['png','svg','pdf']:fig.savefig(O/f'final_continuous_comparison.{ext}',dpi=400,facecolor='white')
plt.close(fig)
bs=json.loads((G/'docs/nifty_test_results/paired_intervals.json').read_text());fig,ax=plt.subplots(figsize=(5.4,2.6));trained=next(x for x in rows if x['interval']=='test_combined' and x['fee_bps']==10 and x['policy']=='seed42_64')['return_total'];bn=['untrained','buy_hold_equal','inverse_vol_rebalanced'];bl=['Untrained','Buy-and-hold','Inverse volatility'];
for i,(name,label) in enumerate(zip(bn,bl)):
 baseline=next(x for x in rows if x['interval']=='test_combined' and x['fee_bps']==10 and x['policy']==name)['return_total'];point=(trained-baseline)*100;lo,hi=next(x for x in bs if x['interval']=='test_combined' and x['baseline']==name and x['block']==20)['return_difference_pp_95CI'];ax.errorbar(point,i,xerr=[[point-lo],[hi-point]],fmt='o',color='#3569A8',capsize=3,markersize=5)
ax.axvline(0,color='#777777',ls='--',lw=.8);ax.set_yticks(range(3),bl);ax.invert_yaxis();ax.set_xlabel('Trained − baseline net return (percentage points)');ax.tick_params(axis='x',length=3);ax.grid(axis='x',color='#eeeeee',lw=.5);fig.tight_layout()
for ext in ['png','svg','pdf']:fig.savefig(O/f'conditional_return_intervals.{ext}',dpi=400,facecolor='white')
plt.close(fig)
(O/'CAPTIONS.md').write_text('''# Final figure captions

Figure1. Net return, zero-risk-free Sharpe, maximum drawdown and annualized daily volatility for the selected InternVL3-2B Nifty native RL adapter (seed42, checkpoint44), original untrained model policy and equal-weight buy-and-hold. Continuous retrospective test1January2025–30January2026,270 observations;100,000 initial cash; identical49-stock panel;10bp purchases-plus-sales fees, funded from account. No year-end reset or terminal liquidation. Training2019–2021; validation2022. Development scores are excluded. Rebalanced equal weight returns13.4406%, exceeding trained13.2829%; additional controls are in the main table. Point estimates are not evidence of established superiority over holding. No market-impact/slippage/tax model. Dataset/universe/preprocessing and retrospective-search limitations remain.

Figure2. Trained seed42 net-return differences and paired circular-block95% intervals over the same continuous test,20-session blocks,5000 resamples. Intervals average per-seed differences while the shown deployment point uses seed42; seed variation is negligible but not treated as independent market observations. Zero denotes no difference. These are conditional intervals, unadjusted for historical method/candidate search. They support an observed raw-relative benefit but not independent confirmatory significance or established holding superiority. Five/60-session sensitivity is preserved in paired_intervals.json. Inverse volatility is periodically rebalanced, not holding.
''',encoding='utf-8')
print('Generated continuous-test publication figures')
