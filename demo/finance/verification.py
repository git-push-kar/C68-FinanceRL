"""Limited factual checks for model drafts; never composes an answer."""
import re

def numbers(text):
 return {float(x.replace(',','')) for x in re.findall(r'(?<!\w)-?\d[\d,]*(?:\.\d+)?',text)}

def issues(answer,facts,evidence_text,question):
 allowed=numbers(evidence_text)|{0,1,2,3,5,10,20,100,126,252}
 unsupported=[x for x in numbers(answer) if not any(abs(abs(x)-abs(y))<=.0011 for y in allowed)]
 problems=[]
 if unsupported:problems.append('Numbers not supported by the supplied evidence: '+', '.join(map(str,sorted(unsupported)))+'. Remove unsupported claims; do not invent replacement quantities.')
 assets=[h for h in facts['holdings'] if re.search(r'\b'+h['ticker']+r'\b',question,re.I)]
 if len(assets)>1 and re.search(r'profit|p&l|pnl|contribut',question,re.I):
  winner=max(assets,key=lambda h:h['cumulative_market_pnl_before_fees_usd'])
  for ticker in re.findall(r'\b(GLD|SPY|TLT|UUP|XLE)\s+(?:had|has|made|contributed)\s+(?:the\s+)?(?:higher|greater|larger|most|more)\s+(?:profit|contribution)',answer):
   if ticker!=winner['ticker']:
    problems.append('Incorrect numeric comparison. Among the requested holdings, the higher cumulative P&L is '+winner['ticker']+' at '+str(winner['cumulative_market_pnl_before_fees_usd'])+' USD. A smaller negative loss is numerically higher than a larger negative loss. Correct the comparison, preserving the recorded values.')
 if 'final_research_comparison' in facts and re.search(r'minimum variance',question,re.I) and re.search(r'instead|compare|benchmark|model|AI',question,re.I):
  benchmark=facts['final_research_comparison']['minimum_variance_sharpe']
  if not any(abs(x-benchmark)<.0011 for x in numbers(answer)):
   problems.append('The requested benchmark comparison omits the standalone minimum-variance Sharpe '+str(benchmark)+'. Include it and distinguish it from the untrained model. The confidence interval concerns trained versus untrained only.')
 if 'final_research_comparison' in facts:
  c=facts['final_research_comparison']
  if c['minimum_variance_sharpe']>facts['sharpe_zero_risk_free']:
   inverted=re.search(r"trained policy(?:'s Sharpe)? (?:is|has|had) (?:the )?higher(?: observed)?(?: Sharpe)? (?:than|compared to) minimum variance",answer,re.I) or re.search(r'Between[^.]*minimum variance, (?:the )?trained policy has the higher',answer,re.I)
   if inverted:problems.append('Incorrect Sharpe ordering: standalone minimum variance '+str(c['minimum_variance_sharpe'])+' exceeds trained '+str(facts['sharpe_zero_risk_free'])+'. Correct the winner. The interval containing zero concerns trained minus initial; statistical superiority over initial is not established.')
 return problems
