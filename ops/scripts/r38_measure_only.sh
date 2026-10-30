#!/bin/bash
# p329: 只测量不重启（用于 systemd 托管下的最终验收）
set -u
python3 - <<'PY'
import json, time, urllib.request
text = open('/root/ascend-dev/eval_text.txt', encoding='utf-8').read()
rq = urllib.request.Request('http://127.0.0.1:8000/fnx/loss',
                            data=json.dumps({'text': text, 'max_tokens': 200}).encode(),
                            headers={'Content-Type': 'application/json'})
d = json.load(urllib.request.urlopen(rq, timeout=1800))
print("  mean_nll=%.4f  ppl=%.3f  top1=%.1f%%  n=%d" % (
    d['mean_nll'], d['ppl'], d['top1_acc'] * 100, d['n']), flush=True)

URL = 'http://127.0.0.1:8000/v1/chat/completions'
def gen(mx, msg='请写一段约200字的散文'):
    b = json.dumps({'messages': [{'role': 'user', 'content': msg}], 'max_tokens': mx,
                    'temperature': 0, 'stream': True}).encode()
    r = urllib.request.Request(URL, data=b, headers={'Content-Type': 'application/json'})
    t0 = time.perf_counter(); first = None; k = 0
    for line in urllib.request.urlopen(r, timeout=900):
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
    print("  max_tokens=128 出=%dtok TTFT=%.2fs TPOT=%.1f ms" % (k, first, t), flush=True)
PY
