from pathlib import Path
import json,shutil,hashlib,zipfile
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
R=Path.cwd();S=Path('F:/Finance_Final_Graphs_2026-11-16/clean_final_comparison')
r=json.loads((R/'deliverables/finance_rl_native_final_20261116/final_evaluation/seed42.json').read_text());a=r['initial']['metrics'];b=r['trained']['metrics']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.linewidth':.65,'axes.spines.top':False,'axes.spines.right':False,'xtick.major.size':0,'ytick.major.size':3,'svg.fonttype':'none','pdf.fonttype':42})
fig,axs=plt.subplots(1,4,figsize=(7.2,2.65));fig.subplots_adjust(left=.065,right=.99,bottom=.18,top=.77,wspace=.58)
spec=[('return_total',100,'Net return (%)',8,[0,2,4,6,8]),('sharpe_proxy',1,'Sharpe ratio',2.4,[0,.5,1,1.5,2]),('max_drawdown',100,'Max. DD (%)',4.5,[0,1,2,3,4]),('volatility_daily',100*np.sqrt(252),'Ann. vol. (%)',7,[0,2,4,6])]
for i,(ax,(key,f,label,limit,ticks)) in enumerate(zip(axs,spec)):
 values=[a[key]*f,b[key]*f];bars=ax.bar([0,1],values,width=.57,color=['#B6BCC5','#3569A8'],edgecolor='#333333',linewidth=.45,zorder=3)
 ax.set_ylim(0,limit);ax.set_xlim(-.7,1.7);ax.set_yticks(ticks);ax.set_xticks([]);ax.set_title(f'({chr(97+i)}) {label}',fontsize=9,pad=10);ax.tick_params(axis='y',labelsize=8);ax.yaxis.grid(True,color='#E8E8E8',lw=.5,zorder=0)
 for bar,v in zip(bars,values):ax.text(bar.get_x()+bar.get_width()/2,v+limit*.025,f'{v:.2f}',ha='center',va='bottom',fontsize=8)
fig.legend(handles=[Patch(facecolor='#B6BCC5',edgecolor='#333333',linewidth=.45,label='Untrained'),Patch(facecolor='#3569A8',edgecolor='#333333',linewidth=.45,label='Trained')],loc='upper center',bbox_to_anchor=(.5,1.0),ncol=2,frameon=False,fontsize=9,handlelength=1.2)
for ext in ['png','svg','pdf']:fig.savefig(S/f'trained_vs_untrained.{ext}',dpi=400,facecolor='white')
plt.close(fig)
for dest in [R/'deliverables/finance_final_figures_20261116/clean_final_comparison',R/'deliverables/finance_context_transfer_20261116_v4/final_presentation_figures/clean_final_comparison',Path('F:/Polyvalent_Finance_Context_2026-11-16_v4/final_presentation_figures/clean_final_comparison')]:
 for ext in ['png','svg','pdf']:shutil.copy2(S/f'trained_vs_untrained.{ext}',dest/f'trained_vs_untrained.{ext}')
note='\n\nPublication figure revision: the primary trained_vs_untrained figure now has only four metric panels, axes, a shared legend and numeric values. Headline, directional guidance, change annotations and takeaway prose are removed. Use CAPTION.md for interpretation, evaluation dates, seed and uncertainty.\n'
for root in [R/'deliverables/finance_context_transfer_20261116_v4',Path('F:/Polyvalent_Finance_Context_2026-11-16_v4')]:
 p=root/'CONTEXT_TRANSFER.md';p.write_text(p.read_text(encoding='utf-8')+note,encoding='utf-8');(root/'TRANSFER_MANIFEST.json').write_text(json.dumps({p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob('*') if p.is_file() and p.name!='TRANSFER_MANIFEST.json'},indent=2))
root=Path('F:/Polyvalent_Finance_Context_2026-11-16_v4')
with zipfile.ZipFile(str(root)+'.zip','w',compression=zipfile.ZIP_DEFLATED) as z:
 for p in root.rglob('*'):
  if p.is_file():z.write(p,p.relative_to(root).as_posix())
