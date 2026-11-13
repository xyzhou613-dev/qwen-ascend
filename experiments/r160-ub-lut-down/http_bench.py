"""Controlled HTTP TPOT; quality observations do not gate timing collection."""
import hashlib,json,os,statistics,sys,time,urllib.request
from pathlib import Path
P=Path(__file__).resolve().parent;phase=sys.argv[1];base=os.environ.get('BENCH_URL','http://127.0.0.1:19002')
out=P/phase;out.mkdir(exist_ok=True)
def request(path,payload):return urllib.request.Request(base+path,data=json.dumps(payload,ensure_ascii=False).encode(),headers={'Content-Type':'application/json'})
def generate(payload):
 start=time.perf_counter();first=None;text=[];stats=None;done=False
 with urllib.request.urlopen(request('/v1/chat/completions',dict(payload,stream=True)),timeout=900) as response:
  for line in response:
   if not line.startswith(b'data: '):continue
   body=line[6:].strip()
   if body==b'[DONE]':done=True;break
   event=json.loads(body)
   if 'fnx_stats' in event:stats=event['fnx_stats']
   content=event['choices'][0]['delta'].get('content','')
   if content:
    if first is None:first=time.perf_counter()-start
    text.append(content)
 assert done and stats is not None and first is not None,(done,stats)
 return dict(ttft_s=first,end_to_end_s=time.perf_counter()-start,stats=stats,answer=''.join(text))
short=[];long=[]
for i in (1,2,3):
 raw=(P/'inputs'/f'payload{i}.json').read_bytes();r=generate(json.loads(raw));r.update(run=i,payload_sha256=hashlib.sha256(raw).hexdigest())
 assert r['stats']['n_prefill']>=8060 and r['stats']['tokens']==128,('SHORT_TOKEN_COUNT',r['stats'])
 short.append(r);(out/'8k.json').write_text(json.dumps(short,ensure_ascii=False,indent=2))
 print('FRESH_8K',phase,i,round(r['ttft_s'],3),r['stats']['tpot_ms'],flush=True)
for i,name in enumerate(('long-payload.json','long-payload-second.json','long-payload-third.json'),1):
 raw=(P/'inputs'/name).read_bytes();payload=json.loads(raw);r=generate(payload)
 r.update(run=i,payload_sha256=hashlib.sha256(raw).hexdigest(),recall_hits=sum(c in r['answer'] for c in ('7319','8652','2946')))
 assert r['stats']['n_prefill']>=262012 and r['stats']['tokens']==128,('LONG_TOKEN_COUNT',r['stats'])
 long.append(r);(out/'256k.json').write_text(json.dumps(long,ensure_ascii=False,indent=2))
 print('FRESH_256K',phase,i,round(r['ttft_s'],3),r['stats']['tpot_ms'],'recall',r['recall_hits'],flush=True)
 if i==1:
  t=time.perf_counter()
  with urllib.request.urlopen(request('/v1/chat/completions',dict(payload,stream=False)),timeout=180) as resp:cache=json.load(resp)
  cache_result=dict(end_to_end_s=time.perf_counter()-t,stats=cache['fnx_stats'],usage=cache['usage'],answer=cache['choices'][0]['message']['content'])
  cache_result['recall_hits']=sum(c in cache_result['answer'] for c in ('7319','8652','2946'))
  assert cache_result['stats']['n_prefill']==0 and cache_result['usage']['completion_tokens']==128
  (out/'cached256k.json').write_text(json.dumps(cache_result,ensure_ascii=False,indent=2))
payload=json.loads((P/'inputs/payload1.json').read_bytes());payload['max_tokens']=1024
sustained=generate(payload);assert sustained['stats']['n_prefill']>=8060 and sustained['stats']['tokens']>=256,sustained['stats']
(out/'sustained1024.json').write_text(json.dumps(sustained,ensure_ascii=False,indent=2))
print('SUSTAINED_1024',phase,sustained['stats']['tpot_ms'],flush=True)
summary=dict(phase=phase,base_url=base,median_8k_ttft_s=statistics.median(r['ttft_s'] for r in short),
 median_8k_tpot_ms=statistics.median(r['stats']['tpot_ms'] for r in short),
 median_256k_ttft_s=statistics.median(r['ttft_s'] for r in long),median_256k_tpot_ms=statistics.median(r['stats']['tpot_ms'] for r in long),
 cached_256k_tpot_ms=cache_result['stats']['tpot_ms'],sustained1024_tpot_ms=sustained['stats']['tpot_ms'],sustained_actual_tokens=sustained['stats']['tokens'],
 short_tpot_ms=[r['stats']['tpot_ms'] for r in short],long_tpot_ms=[r['stats']['tpot_ms'] for r in long],long_recall_hits=[r['recall_hits'] for r in long])
(out/'summary.json').write_text(json.dumps(summary,indent=2));print('BENCH_COMPLETE',json.dumps(summary),flush=True)
