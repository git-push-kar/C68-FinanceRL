from pathlib import Path
import sys,json,hashlib,shutil
import numpy as np
R=Path.cwd();D=R.parent/'PolyvalentDemo';sys.path.insert(0,str(R/'rl_candidate_static_native'))
from env import PortfolioEnv
p,_,tickers,dates=PortfolioEnv._load_prices(str(R/'data/nifty50_prices.csv'));keep=dates>=np.datetime64('2023-01-01');p=p[keep];dates=dates[keep]
out=D/'data/finance_paper';out.mkdir(exist_ok=True,parents=True);np.savez_compressed(out/'nifty_prices.npz',prices=p,dates=dates.astype('datetime64[D]'),tickers=np.array(tickers))
package=R/'campaign_runs/nifty_static_native_seed42_64/checkpoint_44';dest=D/'adapters/finance_policy';shutil.copytree(package,dest,dirs_exist_ok=True)
meta={'title':'Nifty paper portfolio','currency':'INR','base_capital':100000,'first_return':'2025-01-01','last_return':'2026-01-30','source_sha256':hashlib.sha256((R/'data/nifty50_prices.csv').read_bytes()).hexdigest(),'panel_sha256':hashlib.sha256((out/'nifty_prices.npz').read_bytes()).hexdigest(),'adapter_sha256':hashlib.sha256((dest/'finance_lora/finance/adapter_model.safetensors').read_bytes()).hexdigest(),'adapter':'Nifty native static seed42 checkpoint44','tickers':tickers,'rebalance_sessions':5,'fee_bps':10,'prompt':(R/'rl_candidate_static_native/exposure_prompt.py').read_text(),'recorded_probabilities':json.loads((R/'reports/nifty_final_candidate_test_20261116/protocol.json').read_text())['policies'],'research_note':'Retrospective test. Not a live-market or broker service. Original adjusted-price universe and preprocessing retained.'}
# Store the exact policy prompt as text, not executable code.
from exposure_prompt import prompt
meta['prompt']=prompt(None,tickers)
(out/'manifest.json').write_text(json.dumps(meta,indent=2));shutil.copy2(R/'rl_candidate_static_native/cash_execution.py',D/'finance/cash_execution.py')
print('Prepared pinned policy and original adjusted-price panel',p.shape)
