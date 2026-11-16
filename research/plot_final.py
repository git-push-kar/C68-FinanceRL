"""Render all frozen primary seeds and the matched final-period controls."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--results',required=True)
    p.add_argument('--analysis',required=True);p.add_argument('--output',required=True);args=p.parse_args()
    folder=Path(args.results);output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    analysis=json.loads(Path(args.analysis).read_text())
    if json.loads((folder/'status.json').read_text())['status']!='completed':raise ValueError('Final run incomplete')
    plot=output/'finance_final_2025.png'
    if plot.exists():raise FileExistsError(plot)
    sources={};fig,axes=plt.subplots(2,1,figsize=(12,8),sharex=True,gridspec_kw={'height_ratios':[2,1]})
    methods=[(f'candidate_seed{s}',f'Adapter seed {s}',c,'-') for s,c in zip((42,123,456),('#075985','#0284c7','#38bdf8'))]
    methods += [('minimum_variance252_monthly','Minimum variance','#c2410c','-'),
                ('hrp252_monthly','HRP','#15803d','--'),('equal_weight_daily','Daily equal weight','#6b7280',':')]
    for name,label,color,style in methods:
        path=folder/(name+'.json');r=json.loads(path.read_text());daily=r['daily']
        dates=pd.to_datetime([daily[0]['decision_date']]+[d['return_date'] for d in daily])
        equity=np.r_[1.,np.cumprod([1+d['net_return'] for d in daily])]
        dd=100*(equity/np.maximum.accumulate(equity)-1)
        axes[0].plot(dates,equity,label=label,color=color,linestyle=style,linewidth=1.5)
        axes[1].plot(dates,dd,color=color,linestyle=style,linewidth=1.2)
        sources[name]=hashlib.sha256(path.read_bytes()).hexdigest()
    axes[0].set_title('Frozen post-publication test · May 2025–January 2026 · 10-bp traded-stock L1 costs',loc='left',pad=14)
    axes[0].set_ylabel('Wealth / starting equity');axes[1].set_ylabel('Drawdown (%)')
    axes[0].legend(ncol=3,loc='best',frameon=False,fontsize=9)
    for ax in axes:ax.grid(alpha=.18);ax.spines[['top','right']].set_visible(False)
    axes[1].xaxis.set_major_locator(mdates.MonthLocator(interval=2));axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    fig.text(.08,.01,'One-close signal delay; fixed historical stock universe; risk-free rate = 0. Seeds share one market history. See methods/provenance limits.',fontsize=9,color='#475569')
    fig.tight_layout(rect=(0,.035,1,1));fig.savefig(plot,dpi=180);fig.savefig(output/'finance_final_2025.pdf');plt.close(fig)
    comparisons=[(k,v) for k,v in analysis['paired_comparisons'].items() if v['optimization_replicates']==3]
    labels={
        'matched_initialization':'Matched initialization','trained_no_lora':'Independently trained no-LoRA',
        'numeric_teacher_transformer':'Numeric teacher Transformer','minimum_variance':'Minimum variance',
        'HRP':'HRP','daily_equal_weight':'Daily equal weight','fixed_initial_risk':'Fixed initial-risk target',
        'numeric_direct_utility_trained_zero_lag':'Direct Sharpe (trained zero lag)',
        'numeric_direct_utility_matched_lag':'Direct Sharpe (matched lag)','numeric_direct_risk':'Numeric direct risk',
        'teacher_warmup':'Teacher parent','joint_teacher_reference':'Joint teacher reference'}
    fig,ax=plt.subplots(figsize=(10,6));positions=np.arange(len(comparisons))
    for i,(name,value) in enumerate(comparisons):
        lo,hi=value['block_bootstrap']['20']['percentile_95_interval'];point=value['mean_seed_delta_sharpe']
        ax.plot([lo,hi],[i,i],color='#64748b',linewidth=2);ax.scatter([point],[i],color='#0369a1',s=40,zorder=3)
    ax.axvline(0,color='#b91c1c',linestyle='--',linewidth=1);ax.set_yticks(positions,[labels.get(k,k) for k,_ in comparisons]);ax.invert_yaxis()
    ax.set_xlabel('Mean seed-wise ΔSharpe: primary adapter minus control')
    ax.set_title('Post-publication test · 20-session circular blocks · individual 95% intervals',loc='left',fontsize=11)
    ax.spines[['top','right']].set_visible(False);ax.grid(axis='x',alpha=.2);fig.tight_layout()
    fig.savefig(output/'finance_final_uncertainty.png',dpi=180);fig.savefig(output/'finance_final_uncertainty.pdf');plt.close(fig)
    (output/'finance_final_2025.sources.json').write_text(json.dumps({'result_hashes':sources,
        'protocol_sha256':analysis['protocol_sha256'],'analysis_sha256':hashlib.sha256(Path(args.analysis).read_bytes()).hexdigest()},indent=2))
    print(json.dumps({'equity_plot':str(plot),'uncertainty_plot':str(output/'finance_final_uncertainty.png')},indent=2))


if __name__=='__main__':main()
