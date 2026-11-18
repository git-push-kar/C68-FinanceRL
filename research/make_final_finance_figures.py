from pathlib import Path
import json,csv,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
R=Path.cwd();B=R/'deliverables/finance_rl_native_final_20261116';A=R/'additional_frozen_adapter_audit_20261116';O=R/'deliverables/finance_final_figures_20261116'
O.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'axes.titleweight':'bold','savefig.facecolor':'white','axes.axisbelow':True})
colors=['#64748b','#2563eb','#d97706','#059669']; sources=[]
def read(p):
 sources.append(p);return json.loads(p.read_text())
seeds=[read(B/f'final_evaluation/seed{s}.json') for s in [42,123,456]];ctrl=read(B/'final_evaluation/controls.json')
series=[('Untrained',seeds[0]['initial']),('Trained (seed42)',seeds[0]['trained']),('Minimum variance',ctrl['causal_minimum_variance']),('Equal weight',ctrl['equal_weight'])]
def save(fig,name):
 for ext in ['png','svg','pdf']:fig.savefig(O/f'{name}.{ext}',dpi=240,bbox_inches='tight')
 plt.close(fig)
fig,axes=plt.subplots(2,3,figsize=(15,8.3));spec=[('return_total',100,'Net return (%) ↑'),('sharpe_proxy',1,'Sharpe (zero risk-free) ↑'),('max_drawdown',100,'Maximum drawdown (%) ↓'),('volatility_daily',100*np.sqrt(252),'Annualized volatility (%) ↓'),('mean_turnover',100,'Mean daily turnover (%) ↓'),('fees',1,'Total modeled fees ($) ↓')]
rows=[]
for label,r in series:
 m=dict(r['metrics']);prev=100000.;fees=0
 for d in r['daily']:fees+=prev*d['trading_cost'];prev=d['equity']
 m['fees']=fees;rows.append({'policy':label,**m})
for ax,(key,factor,title) in zip(axes.flat,spec):
 vals=[r[key]*factor for r in rows];bars=ax.bar(range(4),vals,color=colors,width=.68)
 ax.set_xticks(range(4),['Untrained','Trained','Min. variance','Equal weight'],rotation=15,fontsize=9);ax.set_title(title,fontsize=12);ax.grid(axis='y',alpha=.16);ax.set_ylim(0,max(vals)*1.23)
 for bar,v in zip(bars,vals):ax.text(bar.get_x()+bar.get_width()/2,v+max(vals)*.025,f'{v:.2f}',ha='center',fontsize=10)
fig.suptitle('Frozen final evaluation: return, risk and trading costs',fontsize=19,y=1.02)
fig.text(.5,-.035,'2 Feb–30 Oct 2026 · 189 sessions · $100,000 start · GLD / SPY / TLT / UUP / XLE\nDeployment seed42 fixed before testing; trained three-seed mean Sharpe 1.95146. Paired ΔSharpe 95% CI [−0.080, 0.865] includes zero.',ha='center',fontsize=10)
fig.tight_layout();save(fig,'01_final_metrics')
fig,(ax,dd)=plt.subplots(2,1,figsize=(13.5,8),sharex=True,gridspec_kw={'height_ratios':[2,1]})
for (name,r),color in zip(series,colors):
 dates=np.array(['2026-01-30']+[d['return_date'] for d in r['daily']],dtype='datetime64[D]');equity=np.r_[100000,[d['equity'] for d in r['daily']]];draw=(equity/np.maximum.accumulate(equity)-1)*100
 ax.plot(dates,equity,color=color,label=name,lw=2 if 'Trained' in name else 1.5);dd.plot(dates,draw,color=color,lw=1.5)
ax.set_title('Final-period portfolio paths — identical dates, assets and execution',fontsize=17);ax.set_ylabel('Portfolio value ($)');ax.yaxis.set_major_formatter(FuncFormatter(lambda x,p:f'{x/1000:.0f}k'));ax.legend(ncol=2,frameon=False);dd.set_ylabel('Drawdown (%)');ax.grid(alpha=.16);dd.grid(alpha=.16)
fig.text(.5,.005,'Historical simulation, net of 10 bp per dollar bought or sold; five-session rebalance. No modeled slippage, market impact or taxes.',ha='center',fontsize=10);fig.tight_layout(rect=(0,.025,1,1));save(fig,'02_final_equity_drawdown')
history=read(A/'summary.json');periods=[2022,2023,2024,2025,2026];fig,axes=plt.subplots(1,3,figsize=(16,5.8));x=np.arange(5)
for ax,key,factor,title in zip(axes,['return_total','sharpe_proxy','max_drawdown'],[100,1,100],['Net return (%) ↑','Sharpe ↑','Maximum drawdown (%) ↓']):
 for i,(name,c) in enumerate(zip(['initial','trained_seed42'],colors[:2])):
  vals=[next(r[key] for r in history if r['year']==y and r['policy']==name)*factor for y in periods[:-1]]+[seeds[0]['initial' if i==0 else 'trained']['metrics'][key]*factor]
  bars=ax.bar(x+(i-.5)*.34,vals,.34,color=c,label=['Untrained','Trained seed42'][i]);ax.bar_label(bars,fmt='%.2f',padding=3,fontsize=9)
 ax.set_xticks(x,['2022\nDev.','2023\nDev.','2024\nDev.','2025*\nAdditional','2026†\nReserved']);ax.set_title(title);ax.grid(axis='y',alpha=.16);ax.set_ylim(0,ax.get_ylim()[1]*1.15);ax.axvline(2.5,color='#94a3b8',ls=':',lw=1)
axes[0].legend(frameon=False,fontsize=10);fig.suptitle('Same frozen adapter across market periods: mixed return improvement',fontsize=18,y=1.03)
fig.text(.5,.01,'Each annual portfolio resets to cash; these bars are not a continuous investment history.\n2022–24: adaptively reused development. *10 Jan–31 Dec 2025: additional historical audit, previously accessed data. †2 Feb–30 Oct 2026: original reserved test.\nNo retraining. Different period lengths: total-return bars describe each period and are not annualized rankings.',ha='center',fontsize=10);fig.tight_layout(rect=(0,.17,1,.96));save(fig,'03_period_robustness')
with (O/'final_metrics.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
allrows=history+[{'year':2026,'policy':p,**seeds[0][key]['metrics']} for p,key in [('initial','initial'),('trained_seed42','trained')]]
with (O/'period_metrics.csv').open('w',newline='') as f:
 keys=list(dict.fromkeys(k for r in allrows for k in r));w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(allrows)
(O/'trained_seed_replications.json').write_text(json.dumps([{'seed':r['seed'],'metrics':r['trained']['metrics']} for r in seeds],indent=2))
(O/'SOURCE_MANIFEST.json').write_text(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in set(sources)},indent=2))
print(O)
