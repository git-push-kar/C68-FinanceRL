"""Reproducible exploratory learning curves; no held-out-test claim."""
import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);a=p.parse_args();run=Path(a.run)
    cs=json.loads((run/'candidates.json').read_text());selection=json.loads((run/'selection.json').read_text())
    chosen=selection['selected']['step'];steps=[c['step'] for c in cs]
    fig,axes=plt.subplots(1,2,figsize=(11,4.2),layout='constrained')
    for ax,key,title in zip(axes,['validation','development'],['Validation: selects checkpoint','Reused development: diagnostic only']):
        values=[c[key]['sharpe_proxy'] for c in cs]
        ax.plot(steps,values,'o-',color='#2468a0',label='RL policy',markersize=4)
        ax.axhline(values[0],color='#777777',linestyle='--',label='Same policy before RL')
        ax.axvline(chosen,color='#9e5b25',linestyle=':',label=f'Validation selection: {chosen}')
        if key=='development' and (run/'baseline_equal_weight.json').exists():
            baseline=json.loads((run/'baseline_equal_weight.json').read_text())['metrics']['sharpe_proxy']
            ax.axhline(baseline,color='#39825b',linestyle='-.',label='Matched equal weight')
        m=cs[0][key]
        ax.set_title(title+'\n'+m['first_return_date']+' to '+m['last_return_date'],fontsize=10)
        ax.set_xlabel('Recorded training iteration / update');ax.set_ylabel('Annualized net Sharpe (risk-free rate 0)')
        ax.grid(alpha=.2);ax.legend(fontsize=8,loc='best')
    fig.suptitle('Finance RL: one training seed, all recorded checkpoints',fontsize=12)
    fig.savefig(run/'development_learning_curve.png',dpi=180)
    fig.savefig(run/'development_learning_curve.svg')
    print(run/'development_learning_curve.png')

if __name__=='__main__':main()
