from pathlib import Path
import json,copy,numpy as np,hashlib
from finance.paper_engine import Panel,PaperAccount,STRATEGIES
P=Panel();A=PaperAccount(P);q=np.array(P.manifest['recorded_probabilities']['seed42_64']);snaps={}
for i in range(270):
 A.step(lambda:(q,{'mode':'verified_reference_for_assessment'}))
 if A.n in [1,51,101,249,270]:snaps[A.n]=copy.deepcopy(A.snapshot())
cases=[]
def add(n,question,expected,stock=None):
 s=snaps[n];event=s['activities'][-1];i=P.first+event['id']*5-1
 facts={'as_of':s['date'],'currency':'INR','account_value':round(s['equity'],2),'net_gain':round(s['equity']-s['capital'],2),'net_return_pct':round(s['net_return']*100,3),'cumulative_fees_already_deducted':round(s['fees'],2),'maximum_drawdown_pct':round(s['maximum_drawdown']*100,3),'selected_execution_date':event['execution_date'],'signal_cutoff':event['signal_date'],'selected_event_fee':round(event['fee'],2),'strategy_probabilities_pct':{k:round(v*100,3) for k,v in zip(STRATEGIES,event['probabilities'])},'mechanism':'The RL model learned FIXED probabilities over four external allocation strategies. Causal strategy targets change from lagged prices; rebalance every five trading sessions. Targets and trades are computed by the execution engine. No news/fundamentals are used. No separate realised gain per transaction is measured. Current cumulative holding P&L is before portfolio fees, not a trade amount or realised tax gain. Portfolio value already includes modeled fees. Evidence is not an internal chain of thought or a forecast. There is no broker or live market feed.','initial_cash_purchase':event['id']==0}
 names=[stock] if stock else [h['ticker'] for h in sorted(s['holdings'],key=lambda x:x['value'],reverse=True)[:2]]
 if 'Compare' in question or 'loss' in question:
  names=[h['ticker'] for h in s['holdings'] if h['ticker'].replace('.NS','') in question]
 facts['stocks']=[]
 for name in names:
  idx=P.tickers.index(name);h=next(h for h in s['holdings'] if h['ticker']==name);trade=next((t for t in event['trades'] if t['ticker']==name),None);px=P.prices
  facts['stocks'].append({'ticker':name,'current_value':round(h['value'],2),'current_weight_pct':round(h['weight']*100,3),'cumulative_market_pnl_before_fees':round(h['cumulative_market_pnl'],2),'selected_trade_signed_amount':round(trade['amount'],2) if trade else 0,'previous_weight_pct':round(event['previous_weights'][idx]*100,3),'selected_target_weight_pct':round(event['target_weights'][idx]*100,3),'allocation_contributions_percentage_points':{k:round(v*100,3) for k,v in zip(STRATEGIES,np.array(event['strategy_contributions'])[:,idx])},'lagged_annual_volatility_pct':round(np.std(px[i-125:i+1,idx]/px[i-126:i,idx]-1)*np.sqrt(252)*100,3)})
 cases.append({'id':f'q{len(cases):02}','n':n,'question':question,'facts':facts,'manual_expected':expected})
for n in [1,51,101,249]:
 e=snaps[n]['activities'][-1];buy=max(e['trades'],key=lambda t:t['amount']);sell=min(e['trades'],key=lambda t:t['amount']);stock=buy['ticker'];symbol=stock.replace('.NS','')
 add(n,f'Why did we buy {symbol}, and how much did we buy?','Explain recorded buy, target reconciliation, exact amount; no invented company/news motive.',stock)
 add(n,f'Why did we sell {symbol}?','Correct the false premise: recorded buy, not sell.',stock)
 add(n,'What did this rebalance cost? Is that also the total cost so far?','Distinguish selected-event fee from cumulative fees; initial event equal is permissible.')
 if sell['amount']<0:add(n,f'Why did we sell {sell["ticker"].replace(".NS","")}?','Recorded sell due to target versus drift; correct amount if provided.',sell['ticker'])
 else:add(n,'Why did we buy so many stocks at once?','Initial cash investment across49stocks, not signal of a market rally.')
add(101,'Does the 64.49% preference for inverse volatility mean 64.49% of my money went into one stock?','No; strategy preference is distinct from asset allocation.')
add(101,'Does a strategy contribution of one percentage point mean the stock earned a one percent return?','No; allocation contribution is not return.')
add(101,'Do the learned strategy probabilities change every trading day?','No; fixed learned probabilities; external strategy targets change.')
add(101,'Why are current holding weights different from the selected targets?','Price drift and different timestamps, no unscheduled trade implied.')
add(270,'Should I subtract cumulative trading fees from the portfolio balance to get the true balance?','No; already included, subtracting again double-counts.')
add(270,'Did the model read company news before making these trades?','No; no news input or live feed in policy.')
add(270,'What will this portfolio return next month?','Cannot establish forecast from evidence; no invented guarantee.')
add(270,'Can you prove the alternative trade would have lost money?','No counterfactual evidence supplied; do not claim proof.')
neg=sorted([h for h in snaps[270]['holdings'] if h['cumulative_market_pnl']<0],key=lambda h:h['cumulative_market_pnl']);x,y=neg[0],neg[-1]
add(270,f'Compare {x["ticker"].replace(".NS","")} and {y["ticker"].replace(".NS","")}. Which contributed more market profit?','Less-negative P&L is higher; both losses, not profits from latest trades.')
add(270,f'Is the loss on {x["ticker"].replace(".NS","")} larger than the loss on {y["ticker"].replace(".NS","")}?','Yes; magnitude of the more-negative loss is larger. Correct yes/no and ordering.')
stock=snaps[101]['activities'][-1]['trades'][0]['ticker'];sym=stock.replace('.NS','')
add(101,f'How much {sym} do I own now, and is its market P&L the amount traded?', 'Correct current value and distinguish cumulative P&L from selected trade.',stock)
add(101,f'Was the selected {sym} trade profitable on its own?','Not separately tracked; do not equate cumulative holding P&L with individual trade gain.',stock)
add(101,f'What exact information date was available when we traded {sym}?','Correct signal cutoff before execution, not the current portfolio date.',stock)
add(101,'Did you actually send these orders to my broker?','No; simulated historical ledger only.')
O=Path('finance/evaluation/nifty_qa');O.mkdir(exist_ok=True);(O/'cases.json').write_text(json.dumps(cases,indent=2));protocol={'cases':len(cases),'case_sha256':hashlib.sha256((O/'cases.json').read_bytes()).hexdigest(),'candidates':['selected_explanation_SFT_fp32','pristine_base_fp32'],'evaluation':'Fresh wording on actual Nifty decision records, zero history; same short system prompt and facts for both. No retraining. Assistant manual semantic/factual review, not independent blinded evaluation.','gate':'At least90% full passes and no material trade-direction, accounting or fabricated-motive error on these common questions. If no candidate passes, remove conversation and retain only explicit ledger facts. Passing this small audit would not establish unrestricted reliability.','generation':'Greedy, max160newtokens; no answer templates, no automatic correction loop or numeric-reference recall presented as accuracy.'};(O/'protocol.json').write_text(json.dumps(protocol,indent=2));print('Prepared',len(cases),'cases before generation')
