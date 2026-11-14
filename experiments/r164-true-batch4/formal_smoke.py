"""Verify restored production against the existing exact 8k r152 fixture."""
import hashlib,json,time,urllib.request
from pathlib import Path
P=Path(__file__).resolve().parent
R=Path('/models/fnx-diagnostics/r162-w8a8-head-quality-20261114')
payload=json.loads((R/'inputs/payload1.json').read_text())
reference=json.loads((R/'rollback/8k.json').read_text())[0]['answer']
req=urllib.request.Request('http://127.0.0.1:9002/v1/chat/completions',
    data=json.dumps(payload).encode(),headers={'Content-Type':'application/json','Connection':'close'})
start=time.perf_counter();first=None;pieces=[];stats=None;done=False
with urllib.request.urlopen(req,timeout=120) as response:
    for line in response:
        if not line.startswith(b'data: '):continue
        raw=line[6:].strip()
        if raw==b'[DONE]':done=True;break
        event=json.loads(raw)
        if 'fnx_stats' in event:stats=event['fnx_stats']
        content=event['choices'][0]['delta'].get('content','')
        if content:
            if first is None:first=time.perf_counter()
            pieces.append(content)
answer=''.join(pieces)
assert done and stats and stats['tokens']==128 and stats['n_prefill']>=8060
assert answer==reference,'RESTORED_8K_ANSWER_CHANGED'
result=dict(status='passed',stats=stats,answer_matches_r152_baseline=True,
    answer_sha256=hashlib.sha256(answer.encode()).hexdigest(),
    ttft_s=first-start,wall_s=time.perf_counter()-start)
(P/'formal-smoke.json').write_text(json.dumps(result,indent=2))
print('FORMAL_R152_EXACT_8K_PASS',stats['tpot_ms'])
