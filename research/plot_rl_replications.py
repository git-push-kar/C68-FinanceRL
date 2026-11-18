"""Plot all three matched seeds, including non-improvements."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).parent/'reports'
full=json.loads((ROOT/'internvl_ppo_seed_comparison_budget128.json').read_text())
small=json.loads((ROOT/'numeric_control_calibrated_seed_comparison.json').read_text())
fig,axes=plt.subplots(1,2,figsize=(10.5,4.7),layout='constrained')
colors=['#27669b','#b2642c'];labels=['InternVL + encoder/head PPO','Calibrated small-network PPO']
for offset,result,color,label in zip([-.17,.17],[full,small],colors,labels):
    axes[0].bar(np.arange(3)+offset,result['individual_Sharpe_gains'],width=.32,color=color,label=label)
axes[0].set_xticks(range(3),['Seed 42','Seed 123','Seed 456'])
axes[0].set_ylabel('Change in net Sharpe from identical initialization')
axes[0].set_title('Every training seed; matched 128-iteration budget',fontsize=10)
for i,(result,color) in enumerate(zip([full,small],colors)):
    mean=result['mean_Sharpe_gain'];low,high=result['paired_95_intervals']['20']['shared_time_blocks_and_resampled_seeds']
    axes[1].errorbar(i,mean,yerr=[[mean-low],[high-mean]],fmt='o',color=color,capsize=6)
    axes[1].annotate(f'{mean:+.3f}',(i,mean),xytext=(10,0),textcoords='offset points',va='center')
axes[1].set_xticks([0,1],['InternVL PPO','Small-network PPO'])
axes[1].set_xlim(-.5,1.6);axes[1].set_title('Mean gain and paired 95% bootstrap interval',fontsize=10)
for ax in axes:
    ax.axhline(0,color='#555555',linewidth=1);ax.grid(axis='y',alpha=.2)
axes[0].legend(fontsize=8,loc='upper right')
fig.suptitle('Finance RL development evidence: modest gains, unresolved uncertainty',fontsize=12)
fig.supxlabel('Nifty49, 2023–2024 reused development; checkpoint selection on 2022 only.\nIntervals resample shared 20-day blocks and training seeds; they do not correct method search.',fontsize=9)
fig.savefig(ROOT/'matched_ppo_replications.png',dpi=180)
fig.savefig(ROOT/'matched_ppo_replications.svg')
print(ROOT/'matched_ppo_replications.png')
