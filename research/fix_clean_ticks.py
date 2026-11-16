from pathlib import Path
p=Path('make_clean_finance_comparison.py');s=p.read_text();s=s.replace("ax.set_yticks(np.linspace(0,limit,4));ax.set_yticklabels([f'{v:g}' for v in np.linspace(0,limit,4)])", "ticks={'return_total':[0,2,4,6,8],'sharpe_proxy':[0,.5,1,1.5,2],'max_drawdown':[0,1,2,3,4],'volatility_daily':[0,2,4,6]}[key];ax.set_yticks(ticks);ax.set_yticklabels([f'{v:g}' for v in ticks])")
p.write_text(s)
