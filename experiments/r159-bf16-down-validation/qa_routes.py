import asyncio,hashlib,json,os,uuid
from pathlib import Path
import torch
P=Path(os.environ['R157_DIR']);Q=Path('/models/fnx-diagnostics/r156-moe-kmajor-20261113/private-quality')
def install(namespace):
 app=namespace['app'];Request=namespace['Request'];get_engine=namespace['get_engine'];LOCK=namespace['LOCK'];REQUEST_LOCK=namespace['REQUEST_LOCK']
 from quality_core import score_ids
 @app.post('/__r157_score')
 async def score(req:Request):
  body=await req.json();eng=get_engine()
  if 'natural' in body:
   case=json.load(open(Path('/models/fnx-diagnostics/r157-combined-quality-20261113/private-natural/inputs.json')))[int(body['natural'])]
   ids=case['ids'];scored=case['scored_tail'];name=case['name']
  elif 'case' in body:
   cases=json.load(open(Q/'nll-inputs.json'))['cases'];case=cases[int(body['case'])]
   ids=case['ids'];scored=case['scored_tail'];name=case['name']
  elif body.get('pilot'):
   text='A reliable inference service needs reproducible measurements. We compare the same inputs, record latency and accuracy, and check whether the result survives repeated tests.'
   ids=eng.tok.encode(text)[:17];scored=16;name='pilot16'
  else:
   fixtures=json.load(open('/models/fnx-diagnostics/r152-state-prefill-20261113/fixtures.json'))
   name=body['fixture'];ids=fixtures[name];scored=199
  assert all(isinstance(i,int) and 0<=i<248320 for i in ids) and len(ids)<=262144
  def work():
   with LOCK,torch.no_grad():
    eng.m.id=uuid.uuid4().hex
    r=score_ids(eng.m,ids,scored,body.get('mode','decode'),bool(body.get('legacy',False)))
    r.update(name=name,ids_sha256=hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest(),variant=os.environ['R157_PHASE'])
    return r
  async with REQUEST_LOCK:return await asyncio.to_thread(work)
 @app.post('/__r157_audit')
 async def audit(req:Request):
  eng=get_engine()
  def work():
   with LOCK:
    from fnx import audit as au
    return dict(parent=au.status(),worker=eng.m._rpc('qa_diag'),variant=os.environ['R157_PHASE'])
  async with REQUEST_LOCK:return await asyncio.to_thread(work)
