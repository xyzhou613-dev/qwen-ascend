#!/bin/bash
# p339_prod.sh —— 生产验收：NLL 指标 + MCQ 24 题 + 生成速度（在宿主机上跑，打 127.0.0.1:8000）
set -u
python3 - <<'PY'
import json, time, urllib.request

BASE = 'http://127.0.0.1:8000'

def post(path, payload, timeout=3600):
    rq = urllib.request.Request(BASE + path, data=json.dumps(payload).encode(),
                                headers={'Content-Type': 'application/json'})
    return json.load(urllib.request.urlopen(rq, timeout=timeout))

# 1) NLL 指标
text = open('/root/ascend-dev/eval_text.txt', encoding='utf-8').read()
d = post('/fnx/loss', {'text': text, 'max_tokens': 200})
print("  mean_nll=%.4f  ppl=%.3f  top1=%.1f%%  n=%d" % (
    d['mean_nll'], d['ppl'], d['top1_acc'] * 100, d['n']), flush=True)

# 2) 生成速度
URL = BASE + '/v1/chat/completions'

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

gen(16)
for _ in range(2):
    t, k, first = gen(128)
    print("  max_tokens=128 出%dtok TTFT=%.2fs TPOT=%.1f ms" % (k, first, t), flush=True)
PY
