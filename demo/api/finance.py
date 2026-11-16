from fastapi import APIRouter, HTTPException
from pathlib import Path
from pydantic import BaseModel,Field
from typing import Literal
from finance.service import evidence
from finance.grounding import context
from finance.generation import generate

router=APIRouter(prefix='/finance',tags=['finance'])

@router.get('/replay')
def replay():
    try:return evidence()
    except FileNotFoundError as exc:raise HTTPException(503,str(exc)) from exc

class Turn(BaseModel):
    role:Literal['user','assistant']
    content:str=Field(min_length=1,max_length=8000)

class Question(BaseModel):
    question:str=Field(min_length=1,max_length=1200)
    seed:Literal[42,123,456]=42
    decision_id:int=Field(ge=0)
    capital:float=Field(ge=100,le=10000000,allow_inf_nan=False)
    as_of_index:int|None=Field(default=None,ge=0)
    history:list[Turn]=Field(default_factory=list,max_length=12)

def validate(request):
    deployment = Path(__file__).resolve().parents[1]/'finance/deployment.json'
    if deployment.exists():
        import json
        if not json.loads(deployment.read_text())['explanations']['enabled']:
            raise HTTPException(410, 'Finance conversation was removed after factual-accuracy testing. Use the paper portfolio decision records.')
    if (Path(__file__).resolve().parents[1]/'finance/training_active.json').exists():raise HTTPException(503,'The finance explanation model is being trained/evaluated. Portfolio records remain available; try questions after the run finishes.')
    try:context(request.seed,request.decision_id,request.capital,request.as_of_index,request.question)
    except (KeyError,ValueError) as exc:raise HTTPException(400,str(exc)) from exc
    except FileNotFoundError as exc:raise HTTPException(503,str(exc)) from exc

def events(request):
    from api.server import get_service
    yield from generate(get_service(),request.question,request.seed,request.decision_id,request.capital,
                        request.as_of_index,[x.model_dump() for x in request.history])

@router.post('/explain/stream')
def explain_stream(request:Question):
    validate(request)
    from api.server import _sse_response,_sse
    def stream():
        try:
            for event in events(request):yield _sse(event)
        except Exception as exc:
            yield _sse({'event':'error','text':str(exc),'mode':'generation_failed','preset_fallback':False})
    return _sse_response(stream())

@router.post('/explain')
def explain(request:Question):
    """Nonstreaming compatibility endpoint; also actual model generation."""
    validate(request)
    try:
        output=[];receipt={}
        for event in events(request):
            if event['event']=='token':output.append(event['text'])
            if event['event']=='done':receipt=event
        return {'answer':''.join(output),**receipt}
    except Exception as exc:raise HTTPException(503,f'Model generation failed: {exc}. No preset fallback used.') from exc
