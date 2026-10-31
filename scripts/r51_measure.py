"""Local HTTP end-to-end streaming timings, exact 8064-token fresh prompts."""
import json,time,urllib.request,os
from pathlib import Path

out=Path(os.environ.get('BENCH_DIR','/root/ascend-dev/r51'))
results=[]
for i in range(1,4):
    data=(out/f'payload{i}.json').read_bytes()
    request=urllib.request.Request('http://127.0.0.1:8000/v1/chat/completions',
            data=data,headers={'Content-Type':'application/json'})
    start=time.perf_counter();first=None;stats=None;answer=[]
    with urllib.request.urlopen(request,timeout=1800) as response:
        headers=time.perf_counter()-start
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
    elapsed=time.perf_counter()-start
    assert stats is not None
    result={'run':i,'transport':'server-local HTTP','prompt_tokens':8064,
            'output_tokens':stats['tokens'],'ttft_s':round(first,3),
            'end_to_end_s':round(elapsed,3),'prefill_s':round(stats['prefill_ms']/1000,3),
            'prefill_tok_s':stats['prefill_tok_s'],'fresh_prefill_tokens':stats['n_prefill'],
            'decode_tpot_ms':stats['tpot_ms'],'server_total_s':stats['seconds']}
    results.append(result)
    (out/f'answer{i}.txt').write_text(''.join(answer),encoding='utf-8')
    (out/'results.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False),flush=True)
    assert stats['n_prefill']>=8060,'unexpected prompt cache hit'
print('E2E_COMPLETE',flush=True)
