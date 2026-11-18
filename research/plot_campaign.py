"""Plot actual selected return paths, preserving seed variation and controls."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT=Path(__file__).parent


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prefix',default='covariance_teacher_seed')
    p.add_argument('--output',required=True)
    args=p.parse_args();output=Path(args.output)
    if output.exists():raise FileExistsError(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    series=[];sources=[]
    for seed,color in zip((42,123,456),('#075985','#0284c7','#38bdf8')):
        folder=ROOT/'campaign_runs'/f'{args.prefix}{seed}'
        selection=json.loads((folder/'selection.json').read_text())
        step=selection['selected']['step']
        path=folder/f'development_{step}.json'
        record=json.loads(path.read_text())
        series.append((f'InternVL seed {seed}',record['daily'],color,1.4,.9))
        sources.append(str(path))
    controls=ROOT/'campaign_runs/classical_development_20261107'
    for label,name,color,style in [('Numerical minimum variance','minimum_variance252_monthly','#222222','-'),
        ('Hierarchical risk parity','hrp252_monthly','#8f5488','-'),
        ('Daily equal weight','equal_weight_daily','#8b9095','--')]:
        path=controls/(name+'.json');record=json.loads(path.read_text())
        series.append((label,record['daily'],color,2.,1.));sources.append(str(path))
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,(top,bottom)=plt.subplots(2,1,figsize=(12,7),sharex=True,gridspec_kw={'height_ratios':[2,1]})
    for label,infos,color,width,alpha in series:
        returns=np.array([i['net_return'] for i in infos]);dates=pd.to_datetime([i['return_date'] for i in infos])
        wealth=100*np.cumprod(1+returns)
        peak=np.maximum.accumulate(np.r_[100.,wealth])[1:]
        style='--' if label=='Daily equal weight' else '-'
        top.plot(dates,wealth,label=label,color=color,lw=width,alpha=alpha,ls=style)
        bottom.plot(dates,100*(wealth/peak-1),color=color,lw=width,alpha=alpha,ls=style)
    top.set_ylabel('Portfolio value (initial = 100)');bottom.set_ylabel('Drawdown (%)')
    for axis in (top,bottom):axis.grid(axis='y',alpha=.18);axis.margins(x=.01)
    top.legend(loc='upper left',ncols=2,frameon=False,fontsize=9)
    top.set_title('Covariance-conditioned InternVL finance: selected development portfolios',loc='left',pad=12)
    bottom.set_xlabel('Return date')
    fig.text(.08,.015,'2023–2024 reused development data · 49-stock fixed panel · 10 bps × absolute traded weight · three seeds, one market history',fontsize=9,color='#555555')
    fig.tight_layout(rect=(0,.035,1,1));fig.savefig(output,dpi=180)
    output.with_suffix('.sources.json').write_text(json.dumps(sources,indent=2))
    print(str(output.resolve()))


if __name__=='__main__':main()
