from pathlib import Path
import sys,json,hashlib,shutil,zipfile,csv
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
R=Path.cwd();B=R/'deliverables/finance_rl_native_final_20261116';S=Path('F:/Finance_Final_Graphs_2026-11-16/clean_final_comparison');O=R/'buy_and_hold_audit_20261116';O.mkdir(exist_ok=True)
sys.path.insert(0,str(B/'runtime_source'));from metrics import metrics_from_infos
r=json.loads((B/'final_evaluation/seed42.json').read_text());ref=r['initial']['daily'];source=B/'native_final_data_20261116/five_asset_final_with_warmup.csv'
f=pd.read_csv(source);p=f.pivot(index='Date',columns='Ticker',values='Close').sort_index();tickers=['GLD','SPY','TLT','UUP','XLE'];p=p[tickers]
assert not p.isna().any().any()
start=ref[0]['decision_date'];dates=[d['return_date'] for d in ref];capital=100000.;fee_rate=.001;fee=capital*fee_rate/(1+fee_rate);notional=capital-fee
units=notional/5/p.loc[start].to_numpy();previous=capital;peak=capital;daily=[];last_prices=p.loc[start].to_numpy()
for i,d in enumerate(ref):
 prices=p.loc[d['return_date']].to_numpy();value=float(units@prices);gross=float((units@prices)/(units@last_prices)-1);net=value/previous-1;peak=max(peak,value)
 daily.append({'net_return':net,'gross_return':gross,'trading_cost':fee/capital if i==0 else 0.,'turnover':notional/capital if i==0 else 0.,'equity':value,'drawdown':1-value/peak,'decision_date':d['decision_date'],'return_date':d['return_date']})
 previous=value;last_prices=prices
metrics=metrics_from_infos(daily);assert len(daily)==189;assert abs(metrics['final_equity']-notional*np.mean(p.loc[dates[-1]].to_numpy()/p.loc[start].to_numpy()))<1e-8
assert sum(x['turnover'] for x in daily[1:])==0
result={'protocol':{'initial_allocation':'20% each GLD/SPY/TLT/UUP/XLE, post-fee','purchase_date':start,'end':dates[-1],'initial_capital':capital,'initial_fee_usd':fee,'fee_bps':10,'rebalance':'never','terminal_liquidation':False,'dividends':'reflected in same vendor-adjusted prices as existing tests','source_sha256':hashlib.sha256(source.read_bytes()).hexdigest()},'metrics':metrics,'daily':daily,'units':dict(zip(tickers,units.tolist()))}
(O/'buy_and_hold.json').write_text(json.dumps(result,indent=2));print(json.dumps(metrics,indent=2))
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.linewidth':.65,'axes.spines.top':False,'axes.spines.right':False,'xtick.major.size':0,'ytick.major.size':3,'svg.fonttype':'none','pdf.fonttype':42})
fig,axs=plt.subplots(1,4,figsize=(7.6,2.8));fig.subplots_adjust(left=.06,right=.99,bottom=.16,top=.76,wspace=.6)
colors=['#B6BCC5','#3569A8','#C58D42'];names=['Untrained','Trained','Buy-and-hold'];rows=[r['initial']['metrics'],r['trained']['metrics'],metrics]
spec=[('return_total',100,'Net return (%)'),('sharpe_proxy',1,'Sharpe ratio'),('max_drawdown',100,'Max. DD (%)'),('volatility_daily',100*np.sqrt(252),'Ann. vol. (%)')]
from matplotlib.ticker import MaxNLocator
for i,(ax,(key,factor,label)) in enumerate(zip(axs,spec)):
 vals=[m[key]*factor for m in rows];bars=ax.bar([0,1,2],vals,width=.62,color=colors,edgecolor='#333333',linewidth=.45,zorder=3)
 ax.set_ylim(0,max(vals)*1.22);ax.set_xlim(-.7,2.7);ax.yaxis.set_major_locator(MaxNLocator(nbins=4,min_n_ticks=3));ax.set_xticks([]);ax.set_title(f'({chr(97+i)}) {label}',fontsize=9,pad=10);ax.tick_params(axis='y',labelsize=8);ax.yaxis.grid(True,color='#E8E8E8',lw=.5,zorder=0)
 for bar,v in zip(bars,vals):ax.text(bar.get_x()+bar.get_width()/2,v+max(vals)*.03,f'{v:.2f}',ha='center',va='bottom',fontsize=8)
fig.legend(handles=[Patch(facecolor=c,edgecolor='#333333',linewidth=.45,label=n) for c,n in zip(colors,names)],loc='upper center',bbox_to_anchor=(.5,1),ncol=3,frameon=False,fontsize=9,handlelength=1.2)
for ext in ['png','svg','pdf']:fig.savefig(S/f'trained_untrained_buyhold.{ext}',dpi=400,facecolor='white')
plt.close(fig)
caption='''Figure. Untrained and trained InternVL3-2B policy (deployment seed42) versus equal-weight buy-and-hold on the same five ETFs. The purchase date is30January2026 and the189 daily returns run2February–30October2026. Buy-and-hold invests20% in each ETF after the initial10bp purchase fee ($99.90 on $100,000), holds fixed adjusted-price units and makes no subsequent trades. The two model policies rebalance every five sessions. All figures use identical adjusted prices and the same cost convention, with no terminal liquidation, taxes, slippage or market impact. Sharpe assumes zero risk-free return; volatility is annualized using sqrt252. Buy-and-hold is a different baseline from periodically rebalanced equal weight. Point estimates shown; see main report for three-seed replication and temporal uncertainty.\n'''
(S/'BUYHOLD_CAPTION.md').write_text(caption,encoding='utf-8');shutil.copy2(O/'buy_and_hold.json',S/'buy_and_hold.json')
with (S/'three_policy_metrics.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=['policy',*rows[0].keys()]);w.writeheader();w.writerows({'policy':n,**m} for n,m in zip(names,rows))
note=f'''\n\n## Buy-and-hold baseline added\n\nThe preferred final comparison is now final_presentation_figures/clean_final_comparison/trained_untrained_buyhold.png (also SVG/PDF). It retains the untrained and trained seed42 policies and adds an equal-weight, initial-purchase-only baseline on the same five ETFs and exact189 return dates. One initial purchase at30January2026 costs ${fee:.2f}; thereafter units remain fixed, with no rebalancing or terminal liquidation. Vendor-adjusted prices reflect dividends on the same basis as existing results. This is not the previously reported periodically rebalanced equal-weight comparator. Buy-and-hold net return {metrics['return_total']*100:.6f}%, Sharpe {metrics['sharpe_proxy']:.6f}, max drawdown {metrics['max_drawdown']*100:.6f}%, annualized volatility {metrics['volatility_daily']*np.sqrt(252)*100:.6f}%. Source hashes, fixed units and daily values are in buy_and_hold.json; caption is BUYHOLD_CAPTION.md. No adapter or original financial bundle was modified. Endpoint-price wealth identity and absence of subsequent turnover were checked.\n'''
for root in [R/'deliverables/finance_context_transfer_20261116_v4',Path('F:/Polyvalent_Finance_Context_2026-11-16_v4')]:
 dest=root/'final_presentation_figures/clean_final_comparison'
 for pattern in ['trained_untrained_buyhold.*','BUYHOLD_CAPTION.md','buy_and_hold.json','three_policy_metrics.csv']:
  for p in S.glob(pattern):shutil.copy2(p,dest/p.name)
 p=root/'CONTEXT_TRANSFER.md';p.write_text(p.read_text(encoding='utf-8')+note,encoding='utf-8');(root/'TRANSFER_MANIFEST.json').write_text(json.dumps({p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob('*') if p.is_file() and p.name!='TRANSFER_MANIFEST.json'},indent=2))
root=Path('F:/Polyvalent_Finance_Context_2026-11-16_v4')
with zipfile.ZipFile(str(root)+'.zip','w',compression=zipfile.ZIP_DEFLATED) as z:
 for p in root.rglob('*'):
  if p.is_file():z.write(p,p.relative_to(root).as_posix())
