"""Real finance-adapter generation on the application's shared model service."""
import time
import os
from finance.grounding import context,messages
from finance.verification import issues

def generate(service,question,seed,decision_id,capital,as_of_index=None,history=None,max_new_tokens=320):
 facts,selected,sources=context(seed,decision_id,capital,as_of_index,question)
 yield {'event':'status','text':'Loading finance model and reading portfolio evidence…','decision_id':selected,'sources':sources}
 started=time.perf_counter()
 with service.lease('finance') as receipt:
  if receipt.get('mode')!='real':raise RuntimeError('Finance requires actual model generation; stub mode is disabled for this endpoint.')
  if service.residency.runner() is None:raise RuntimeError('Finance model is not resident; refusing a placeholder response.')
  yield {'event':'adapter',**receipt}
  tok=service._ensure_tokenizer();history=list(history or [])
  while True:
   prompt=tok.apply_chat_template(messages(question,history,facts),tokenize=False,add_generation_prompt=True)
   input_tokens=len(tok.encode(prompt,add_special_tokens=False))
   if input_tokens<=4096:break
   if not history:raise ValueError('Portfolio evidence exceeds the generation context budget')
   history=history[2:]
  yield {'event':'status','text':'Finance adapter is generating and checking the answer…','input_tokens':input_tokens}
  drafts=[];problems=[]
  for attempt in range(2):
   events=[];text=''
   for event in service.stream(prompt,max_new_tokens=max_new_tokens,temperature=0.,repetition_penalty=1.0,stub=False):
    if event.get('event')=='notice':raise RuntimeError(event.get('text','Generation failed'))
    if event.get('event')=='token':text+=event.get('text','')
    events.append(event)
   if not text.strip():raise RuntimeError('The model returned no text. No preset fallback was used.')
   problems=issues(text,facts,messages(question,history,facts)[-1]['content'],question)
   drafts.append({'attempt':attempt+1,'answer':text,'issues':problems})
   if not problems or attempt==1:break
   yield {'event':'status','text':'Checking figures and asking the model to revise its draft...'}
   revision=messages(question,history,facts)+[{'role':'assistant','content':text},{'role':'user','content':'A numerical evidence check found these problems: '+' '.join(problems)+' Rewrite the complete answer to my original question using the supplied evidence. Do not discuss the check. Do not add new unsupported claims.'}]
   prompt=tok.apply_chat_template(revision,tokenize=False,add_generation_prompt=True)
  # Only generated text is delivered. The check never edits or substitutes prose.
  for event in events:yield event
 yield {'event':'done','mode':'real_model','model':os.environ.get('FINAL_FINANCE_LABEL','InternVL3-2B + finance adapter'),'decision_id':selected,'sources':sources,'elapsed_seconds':round(time.perf_counter()-started,2),'input_tokens':input_tokens,'verification':{'attempts':len(drafts),'unresolved':problems,'scope':'Unsupported numeric values and explicit multi-ticker P&L ordering only; not a full factual guarantee'},'generation_audit':drafts}
