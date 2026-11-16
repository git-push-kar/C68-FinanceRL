from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
R=Path.cwd();B=R/'deliverables/finance_rl_native_final_20261116/final_evaluation';O=Path('F:/Finance_Final_Graphs_2026-11-16/clean_final_comparison');O.mkdir(exist_ok=True)
r=json.loads((B/'seed42.json').read_text());u=r['initial']['metrics'];t=r['trained']['metrics']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'axes.spines.top':False,'axes.spines.right':False,'axes.spines.left':False,'axes.spines.bottom':True,'axes.edgecolor':'#CBD5E1','svg.fonttype':'none','pdf.fonttype':42})
fig,axes=plt.subplots(1,4,figsize=(13.33,7.5));fig.subplots_adjust(left=.055,right=.98,bottom=.27,top=.67,wspace=.42)
fig.text(.055,.91,'Effect of finance RL fine-tuning',fontsize=25,weight='bold',color='#172033')
fig.text(.055,.85,'Final evaluation · 2 February–30 October 2026 · same portfolio and execution rules',fontsize=12,color='#596579')
fig.legend(handles=[Patch(color='#A5AFBD',label='Untrained'),Patch(color='#315BE8',label='Trained')],loc='upper left',bbox_to_anchor=(.045,.81),frameon=False,ncol=2,fontsize=13,handlelength=1.1,columnspacing=2)
spec=[('return_total',100,'Net return','Higher is better',8,'%',2),('sharpe_proxy',1,'Sharpe ratio','Higher is better',2.4,'',2),('max_drawdown',100,'Maximum drawdown','Lower is better',4.5,'%',2),('volatility_daily',100*np.sqrt(252),'Annualized volatility','Lower is better',7,'%',2)]
for ax,(key,f,title,direction,limit,suffix,digits) in zip(axes,spec):
 vals=[u[key]*f,t[key]*f];bars=ax.bar([0,1],vals,width=.55,color=['#A5AFBD','#315BE8'],zorder=3)
 ax.set_ylim(0,limit);ax.set_xlim(-.65,1.65);ax.set_xticks([0,1],['Untrained','Trained'],fontsize=11);ax.tick_params(axis='x',length=0,pad=10);ax.tick_params(axis='y',length=0,labelsize=10,labelcolor='#788393');ticks={'return_total':[0,2,4,6,8],'sharpe_proxy':[0,.5,1,1.5,2],'max_drawdown':[0,1,2,3,4],'volatility_daily':[0,2,4,6]}[key];ax.set_yticks(ticks);ax.set_yticklabels([f'{v:g}' for v in ticks])
 ax.yaxis.grid(True,color='#EDF0F4',linewidth=.9,zorder=0)
 ax.text(0,1.17,title,transform=ax.transAxes,fontsize=14,weight='bold',color='#172033');ax.text(0,1.08,direction,transform=ax.transAxes,fontsize=10.5,color='#687386')
 for b,v in zip(bars,vals):ax.text(b.get_x()+b.get_width()/2,v+limit*.035,f'{v:.2f}{suffix}',ha='center',fontsize=17,weight='bold',color='#172033')
 delta=vals[1]-vals[0];delta_label=f'{delta:+.2f}'+(' pp' if suffix else '')
 ax.text(.5,-.23,delta_label,ha='center',transform=ax.transAxes,fontsize=16,weight='bold',color='#172033')
fig.text(.055,.105,'Lower risk and higher Sharpe; slightly lower net return.',fontsize=17,weight='bold',color='#172033')
fig.text(.055,.055,'Selected deployment adapter (seed 42) · Returns include modeled trading fees · pp = percentage points',fontsize=10,color='#687386')
for ext in ['png','svg','pdf']:fig.savefig(O/f'trained_vs_untrained.{ext}',dpi=240,facecolor='white')
plt.close(fig)
caption='''Figure. Effect of Finance RL fine-tuning on the selected InternVL3-2B deployment adapter (seed42, fixed before final evaluation). Untrained and trained policies were evaluated on the same five ETFs over 189 sessions, 2 February–30 October2026, with identical causal signal timing, five-session rebalancing and 10bp fees per dollar purchased or sold. Net return decreased from6.5318% to6.3770%, while Sharpe increased from1.5131 to1.9514, maximum drawdown decreased from3.5094% to2.4901%, and annualized volatility decreased from5.6835% to4.2717% (daily population standard deviation × sqrt252). Sharpe assumes zero risk-free return. Labels round to two decimals; changes use unrounded values. Lower drawdown and volatility reflect reduced observed risk, not independent demonstrations of superiority. The separately reported three-seed mean paired Sharpe gain has95% interval[-0.080094,0.864912], including zero. Standalone minimum variance scored Sharpe2.1695 and must remain in the paper's baseline table. This figure isolates the requested trained/untrained comparison; it does not establish higher profit or statistical superiority. No slippage, market impact or taxes are modeled.\n'''
(O/'CAPTION.md').write_text(caption,encoding='utf-8')
(O/'metrics.json').write_text(json.dumps({'seed':42,'untrained':u,'trained':t},indent=2))
print(O)
