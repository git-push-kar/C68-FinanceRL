"""Plot fixed validation-selected native candidates; no new portfolio evaluation."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from datetime import datetime

root = Path(__file__).parent
fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex='col')
for col, (title, stem) in enumerate([
    ('Nifty stocks: 2023–2024', 'nifty_static_native'),
    ('Five ETFs: 2022–2024', 'multiasset_native'),
]):
    reference = None
    for seed, color in zip((42, 123, 456), ('#cf6600', '#aa4894', '#278e72')):
        run = root/'campaign_runs'/f'{stem}_seed{seed}_64'
        selection = json.loads((run/'selection.json').read_text())['selected']
        initial = json.loads((run/'development_0.json').read_text())['daily']
        trained = json.loads((run/f'development_{selection["step"]}.json').read_text())['daily']
        dates = [datetime.fromisoformat(str(x['return_date'])[:10]) for x in initial]
        assert [x['return_date'] for x in initial] == [x['return_date'] for x in trained]
        base_returns = np.array([x['net_return'] for x in initial])
        if reference is None:
            reference = base_returns
            series = [('Untrained policy', base_returns, '#243f74', 2.2),
                      (f'RL seed {seed}', np.array([x['net_return'] for x in trained]), color, 1.3)]
        else:
            np.testing.assert_array_equal(reference, base_returns)
            series = [(f'RL seed {seed}', np.array([x['net_return'] for x in trained]), color, 1.3)]
        for label, returns, line_color, width in series:
            equity = np.r_[1., np.cumprod(1 + returns)]
            dd = 100 * (equity / np.maximum.accumulate(equity) - 1)
            axes[0, col].plot(dates, 100*(equity[1:]-1), label=label, color=line_color, lw=width)
            axes[1, col].plot(dates, dd[1:], color=line_color, lw=width)
    axes[0, col].set_title(title)
    axes[0, col].legend(fontsize=8)
    axes[1, col].xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    axes[1, col].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    for ax in axes[:, col]:
        ax.grid(alpha=.2)
        ax.axhline(0, color='#888888', lw=.5)
        ax.spines[['top', 'right']].set_visible(False)
axes[0, 0].set_ylabel('Cumulative net return (%)')
axes[1, 0].set_ylabel('Drawdown (%)')
fig.suptitle('InternVL3-2B finance RL: completed constant-policy candidates', fontsize=14)
fig.text(.5, .015, 'Reused development, not final testing. Each trained seed plotted separately; nearly overlapping lines are not an ensemble.\n'
         'Identical execution within each task; 10 bp × full traded-stock L1. Different tasks are not an absolute-performance ranking.',
         ha='center', fontsize=9)
fig.tight_layout(rect=(0, .07, 1, .95))
for suffix in ('png', 'svg'):
    fig.savefig(root/'reports'/f'native_development_candidates.{suffix}', dpi=180)
print('Wrote reports/native_development_candidates.png and .svg')
