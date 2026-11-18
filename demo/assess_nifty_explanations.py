import os,json,time
from pathlib import Path
from contextlib import nullcontext
D=Path.cwd();os.environ.update(FINAL_FINANCE_TEXT_DTYPE='fp32',FINAL_STUB_GENERATION='0',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
from router.adapter_service import AdapterService
service=AdapterService(base_dir=os.environ.get('FINAL_BASE_DIR',str(D.parent/'models/InternVL3-2B')),finance_dir=str(D/'adapters/finance_explanation'),anomaly_dir='',reasoning_dir='',deepfake_llm_dir='',deepfake_vision_dir='')
O=D/'finance/evaluation/nifty_qa';cases=json.loads((O/'cases.json').read_text());outputs=[]
system='You explain a historical paper portfolio using only the supplied execution record. Answer the exact question in at most three concise sentences. Correct false premises. Distinguish buys from sells, target weights from strategy probabilities, and P&L from traded amounts. Do not invent motives, news, guarantees or unsupported quantities. If the record cannot establish an answer, say so. Currency is INR. Recorded causal signals precede execution; current holdings may have drifted. Give evidence-based interpretation, not hidden chain of thought.'
with service.lease('finance'):
 for name,disabled in [('selected_explanation_SFT_fp32',False),('pristine_base_fp32',True)]:
  with service.residency.model.disable_adapter() if disabled else nullcontext():
   for c in cases:
    messages=[{'role':'system','content':system},{'role':'user','content':'Execution record:\n'+json.dumps(c['facts'])+'\nQuestion: '+c['question']}];prompt=service._ensure_tokenizer().apply_chat_template(messages,tokenize=False,add_generation_prompt=True);started=time.perf_counter();answer='';notices=[]
    for e in service.stream(prompt,max_new_tokens=160,temperature=0,repetition_penalty=1.,stub=False):
     if e['event']=='token':answer+=e['text']
     if e['event']=='notice':notices.append(e['text'])
    r={'candidate':name,'id':c['id'],'answer':answer,'elapsed_seconds':time.perf_counter()-started,'notices':notices};outputs.append(r);(O/'answers.json').write_text(json.dumps(outputs,indent=2));print(name,c['id'],round(r['elapsed_seconds'],2),flush=True)
print('FINISHED',flush=True)
