#!/bin/bash
# p340_tpot.sh —— 只要生成速度（生产热态，绕开 loss 评测）
set -u
python3 - <<'PY'
import json, time, urllib.request
URL = 'http://127.0.0.1:8000/v1/chat/completions'

def gen(mx, msg='请写一段约200字的散文'):
    b = json.dumps({'messages': [{'role': 'user', 'content': msg}], 'max_tokens': mx,
                    'temperature': 0, 'stream': True}).encode()
    rq = urllib.request.Request(URL, data=b, headers={'Content-Type': 'application/json'})
    t0 = time.perf_counter()
    first = None
    k = 0
    for line in urllib.request.urlopen(rq, timeout=900):
        s = line.decode('utf-8', 'ignore').strip()
        if not s.startswith('data: '):
            continue
        p = s[6:]
        if p == '[DONE]':
            break
        e = json.loads(p)
        if e['choices'][0]['delta'].get('content'):
            k += 1
            if first is None:
                first = time.perf_counter() - t0
    tot = time.perf_counter() - t0
    return (tot - (first or 0)) / max(1, k - 1) * 1e3, k, first

gen(8)
for i in range(4):
    t, k, first = gen(128)
    print("  第%d次: %dtok TTFT=%.2fs TPOT=%.1f ms (%.1f tok/s)" % (
        i + 1, k, first, t, 1000.0 / t), flush=True)
PY
