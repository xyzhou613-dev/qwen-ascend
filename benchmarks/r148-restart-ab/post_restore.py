import json,time,urllib.request,hashlib
from pathlib import Path
p=Path('/models/fnx-diagnostics/r148-restart-ab-20261112');payload=json.load(open(p/'inputs/payload1.json'));payload['stream']=False
req=urllib.request.Request('http://127.0.0.1:9002/v1/chat/completions',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
t=time.perf_counter()
with urllib.request.urlopen(req,timeout=180) as f:r=json.load(f)
answer=r['choices'][0]['message']['content'];reference=json.load(open(p/'baseline/8k.json'))[0]['answer']
out=dict(elapsed_s=time.perf_counter()-t,stats=r['fnx_stats'],answer_exact=answer==reference,answer_sha256=hashlib.sha256(answer.encode()).hexdigest())
(p/'post-restore-http.json').write_text(json.dumps(out,indent=2));assert out['answer_exact'] and out['stats']['n_prefill']>=8060;print('POST_RESTORE_HTTP_EXACT',out,flush=True)
