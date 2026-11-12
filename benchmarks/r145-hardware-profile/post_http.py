import hashlib,json,time,urllib.request
from pathlib import Path
P=Path(__file__).resolve().parent;payload=json.load(open(P/'payload8k.json'));payload['stream']=True
req=urllib.request.Request('http://127.0.0.1:9002/v1/chat/completions',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
start=time.perf_counter();first=None;stats=None;answer=[]
with urllib.request.urlopen(req,timeout=180) as f:
 for line in f:
  if not line.startswith(b'data: '):continue
  raw=line[6:].strip()
  if raw==b'[DONE]':break
  event=json.loads(raw)
  if 'fnx_stats' in event:stats=event['fnx_stats']
  content=event['choices'][0]['delta'].get('content','')
  if content:
   if first is None:first=time.perf_counter()-start
   answer.append(content)
answer=''.join(answer);reference=Path('/models/fnx-diagnostics/r144-raw-norm-20261112/production8k/answer1.txt').read_text()
r=dict(ttft_s=first,end_to_end_s=time.perf_counter()-start,stats=stats,answer_exact=answer==reference,answer_sha256=hashlib.sha256(answer.encode()).hexdigest())
P.joinpath('post-restore-http.json').write_text(json.dumps(r,indent=2));assert r['answer_exact'] and stats['n_prefill']>=8060 and stats['tokens']==128,r
print('POST_RESTORE_HTTP_EXACT',round(first,3),stats['tpot_ms'],flush=True)
