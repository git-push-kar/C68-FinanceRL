import json,csv,hashlib,shutil,statistics
from pathlib import Path
R=Path(r'C:\Users\Parth\Documents\PolyvalentFinance');O=R/'deliverables/finance_context_transfer_20261116_v2';O.mkdir(exist_ok=True)
rows=[]
for f in sorted((R/'reports').glob('*comparison*.json')):
 try:
  d=json.loads(f.read_text())
  for r in d.get('runs',[]):
   if not all(isinstance(r.get(k),dict) and 'sharpe_proxy' in r[k] for k in ('initial','trained')):continue
   a,b=r['initial'],r['trained'];rows.append({'report':f.name,'run':r.get('run',''),'selected_step':r.get('selected_step',''),'first':b.get('first_return_date',d.get('dates',{}).get('first','')),'last':b.get('last_return_date',d.get('dates',{}).get('last','')),'initial_sharpe':a['sharpe_proxy'],'trained_sharpe':b['sharpe_proxy'],'delta_sharpe':b['sharpe_proxy']-a['sharpe_proxy'],'initial_return':a.get('return_total'),'trained_return':b.get('return_total'),'initial_drawdown':a.get('max_drawdown'),'trained_drawdown':b.get('max_drawdown')})
 except (ValueError,TypeError):continue
with (O/'ALL_COMPARISON_RUNS.csv').open('w',newline='',encoding='utf-8') as f:
 w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
paths=list((R/'reports').glob('*.json'))+list((R/'reports').glob('*.md'))+list(R.glob('*.md'))+[R/'NATIVE_FINAL_METHOD_CHOICE.json',R/'NATIVE_FINAL_EVALUATION_SPEC.json',R/'reserved_finance_native_2026_access.json']
index=[]
for p in paths:
 index.append({'path':str(p),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
 dest=O/'evidence'/p.relative_to(R);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
(O/'SOURCE_INDEX.json').write_text(json.dumps(index,indent=2),encoding='utf-8')
selected=['multiasset_native_comparison64.json','multiasset_dynamic_native_comparison16.json','readout_only_ablation_comparison16.json','risk_selector_static_comparison64.json','risk_selector_dynamic_comparison64.json','risk_selector_native_comparison64.json','risk_label_rotate_comparison64.json','risk_label_reverse_comparison64.json']
quality=['| Same ETF development history, 2022–2024 | Initial Sharpe | Trained Sharpe (mean) | Trained return | Trained max drawdown |','|---|---:|---:|---:|---:|']
for name in selected:
 rr=[x for x in rows if x['report']==name];mean=lambda k:statistics.mean(x[k] for x in rr)
 quality.append(f"| {name.removesuffix('.json')} | {mean('initial_sharpe'):.6f} | {mean('trained_sharpe'):.6f} | {mean('trained_return'):.3%} | {mean('trained_drawdown'):.3%} |")
(O/'QUALITY_TABLE.md').write_text('\n'.join(quality),encoding='utf-8')
status=json.loads((R/'reports/FINAL_CAMPAIGN_STATUS.json').read_text());ledger=['| Method | Period | Mean ΔSharpe | Positive seeds | Paired 95% interval, 20 sessions | Source |','|---|---|---:|---:|---|---|']
for x in status['results']:
 p=x['period'];ci=x['CI20'];ledger.append(f"| {x['method']} | {p['first']}–{p['last']} | {x['mean_Sharpe_gain']:+.6f} | {x['positive_seeds']}/{x['seeds']} | [{ci[0]:.6f}, {ci[1]:.6f}] | {Path(x['source']).name} |")
(O/'CAMPAIGN_LEDGER.md').write_text('\n'.join(ledger),encoding='utf-8')
B=R/'deliverables/finance_rl_native_final_20261116';replay=json.loads(Path(r'C:\Users\Parth\Documents\PolyvalentDemo\data\finance\replay.json').read_text());controls=json.loads((B/'final_evaluation/controls.json').read_text())
table=['| Policy | Net Sharpe | Total net return | Max drawdown |','|---|---:|---:|---:|'];a=replay['sessions']['42']['initial_metrics']
policies=[('Exact initial native policy',a)]+[(f'Trained seed {seed}',s['metrics']) for seed,s in replay['sessions'].items()]+[('Standalone minimum variance',controls['causal_minimum_variance']['metrics'])]
for name,b in policies:table.append(f"| {name} | {b['sharpe_proxy']:.6f} | {b['return_total']:.4%} | {b['max_drawdown']:.4%} |")
(O/'FINAL_TABLE.md').write_text('\n'.join(table),encoding='utf-8')
print(json.dumps({'source_files':len(index),'comparison_rows':len(rows),'evidence_MB':sum(x['bytes'] for x in index)/1e6}))
