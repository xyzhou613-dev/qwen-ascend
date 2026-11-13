"""Frozen HTTP inputs. No model allocation or tokenizer startup in the client."""
import hashlib,json,os,statistics,sys,time,urllib.request
from pathlib import Path
P=Path(__file__).resolve().parent;phase=sys.argv[1]
base=os.environ.get('BENCH_URL','http://127.0.0.1:19002')
out=P/phase;out.mkdir(exist_ok=True)
def request(path,payload):return urllib.request.Request(base+path,data=json.dumps(payload,ensure_ascii=False).encode(),headers={'Content-Type':'application/json'})
def generate(payload):
    start=time.perf_counter();first=None;answer=[];stats=None
    with urllib.request.urlopen(request('/v1/chat/completions',dict(payload,stream=True)),timeout=900) as response:
        for line in response:
            if not line.startswith(b'data: '):continue
            body=line[6:].strip()
            if body==b'[DONE]':break
            event=json.loads(body)
            if 'fnx_stats' in event:stats=event['fnx_stats']
            content=event['choices'][0]['delta'].get('content','')
            if content:
                if first is None:first=time.perf_counter()-start
                answer.append(content)
    assert stats is not None and first is not None
    return dict(ttft_s=first,end_to_end_s=time.perf_counter()-start,stats=stats,answer=''.join(answer))
records=[]
for i in (1,2,3):
    raw=(P/'inputs'/f'payload{i}.json').read_bytes();payload=json.loads(raw)
    r=generate(payload);r.update(run=i,prompt_tokens=8064,payload_sha256=hashlib.sha256(raw).hexdigest())
    records.append(r);(out/'8k.json').write_text(json.dumps(records,ensure_ascii=False,indent=2))
    assert r['stats']['n_prefill']>=8060 and r['stats']['tokens']==128,('FRESH_8K',r['stats'])
    print('FRESH_8K',phase,i,round(r['ttft_s'],3),r['stats']['tpot_ms'],flush=True)
raw=(P/'inputs/long-payload.json').read_bytes();payload=json.loads(raw)
r=generate(payload);r.update(prompt_tokens=262016,payload_sha256=hashlib.sha256(raw).hexdigest(),recall_hits=sum(code in r['answer'] for code in ('7319','8652','2946')))
(out/'256k.json').write_text(json.dumps(r,ensure_ascii=False,indent=2))
assert r['stats']['n_prefill']>=262012 and r['stats']['tokens']==128 and r['recall_hits']==3,('FRESH_256K',r['stats'],r['recall_hits'])
print('FRESH_256K',phase,round(r['ttft_s'],3),r['stats']['tpot_ms'],flush=True)
start=time.perf_counter()
with urllib.request.urlopen(request('/v1/chat/completions',dict(payload,stream=False)),timeout=180) as response:cached=json.load(response)
repeat=dict(end_to_end_s=time.perf_counter()-start,stats=cached['fnx_stats'],answer=cached['choices'][0]['message']['content'])
(out/'cached256k.json').write_text(json.dumps(repeat,ensure_ascii=False,indent=2))
assert repeat['stats']['n_prefill']==0 and all(code in repeat['answer'] for code in ('7319','8652','2946'))
raw2=(P/'inputs/long-payload-second.json').read_bytes();payload2=json.loads(raw2)
second=generate(payload2);second.update(prompt_tokens=262016,payload_sha256=hashlib.sha256(raw2).hexdigest(),recall_hits=sum(code in second['answer'] for code in ('7319','8652','2946')))
(out/'256k-second.json').write_text(json.dumps(second,ensure_ascii=False,indent=2))
assert second['stats']['n_prefill']>=262012 and second['stats']['tokens']==128 and second['recall_hits']==3,('SECOND_FRESH_256K',second['stats'],second['recall_hits'])
print('SECOND_FRESH_256K',phase,round(second['ttft_s'],3),second['stats']['tpot_ms'],flush=True)
raw3=(P/'inputs/long-payload-third.json').read_bytes();payload3=json.loads(raw3)
third=generate(payload3);third.update(prompt_tokens=262016,payload_sha256=hashlib.sha256(raw3).hexdigest(),recall_hits=sum(code in third['answer'] for code in ('7319','8652','2946')))
(out/'256k-third.json').write_text(json.dumps(third,ensure_ascii=False,indent=2))
assert third['stats']['n_prefill']>=262012 and third['stats']['tokens']==128 and third['recall_hits']==3,('THIRD_FRESH_256K',third['stats'],third['recall_hits'])
print('THIRD_FRESH_256K',phase,round(third['ttft_s'],3),third['stats']['tpot_ms'],flush=True)
summary=dict(phase=phase,base_url=base,median_8k_ttft_s=statistics.median(x['ttft_s'] for x in records),median_8k_tpot_ms=statistics.median(x['stats']['tpot_ms'] for x in records),ttft_256k_s=statistics.median((r['ttft_s'],second['ttft_s'],third['ttft_s'])),ttft_256k_runs_s=[r['ttft_s'],second['ttft_s'],third['ttft_s']],tpot_256k_ms=statistics.median((r['stats']['tpot_ms'],second['stats']['tpot_ms'],third['stats']['tpot_ms'])),tpot_256k_runs_ms=[r['stats']['tpot_ms'],second['stats']['tpot_ms'],third['stats']['tpot_ms']],cached_256k_tpot_ms=repeat['stats']['tpot_ms'])
(out/'summary.json').write_text(json.dumps(summary,indent=2));print('HTTP_BENCH_COMPLETE',summary,flush=True)
