#!/bin/bash
# ★ 第 37 轮：aclnn per-group W4A16（专家路径）生产口径实测
#   基线（同一脚本、同一 config B）：g64 + attn=10 → mean_nll 2.1756 / TPOT 110.3 ms
set -u
CTN=${CTN:-fnx}
PY=/root/ascend-dev/venv/bin/python
LOG=/root/ascend-dev/logs/server.log
RUN=/root/ascend-dev
W4=${W4:-1}

systemctl stop fnx.service >/dev/null 2>&1 || true
pkill -f watchdog.sh 2>/dev/null || true
docker exec $CTN bash -lc "pkill -f '[p]ython -m fnx.server' || true"
for i in $(seq 1 30); do
  docker exec $CTN bash -lc "pgrep -f '[p]ython -m fnx.server'" >/dev/null 2>&1 || break
  sleep 2
done
[ -f "$LOG" ] && mv -f "$LOG" "$LOG.prev"
docker exec -d $CTN bash -lc "cd $RUN/fnx && \
  FNX_PACK=0 FNX_EXP_G=64 FNX_MOE_KERNEL=0 FNX_NO_PLE=1 FNX_PACKED_TORCH=1 \
  FNX_LPC=45 FNX_NLAYERS=48 FNX_PORT=8000 FNX_BATCH=1 \
  FNX_PREFIX_CACHE=1 FNX_CONV_CACHE=4 FNX_PROF=0 FNX_ENABLE_THINKING=0 \
  FNX_DEQ_SMALL=1 FNX_DEQ_HEAD=1 FNX_DEQ_ATTN=10 \
  FNX_LQP=/models/Qwen3.8-Flash-Next-LQP \
  FNX_MOE_W4A16=$W4 \
  $PY -m fnx.server > $LOG 2>&1"
for i in $(seq 1 40); do
  sleep 10
  H=$(curl -s -m 4 http://127.0.0.1:8000/health || true)
  [ -n "$H" ] && { echo "ready: $H"; break; }
  grep -qiE "Tried to allocate|Traceback" "$LOG" 2>/dev/null && {
    echo "FAIL"; grep -iE "Tried to allocate|Error|Traceback" "$LOG" | tail -4; exit 1; }
done
grep -E "hbm0:|装配|图捕获完成" "$LOG" | tail -3
echo "--- 质量（/fnx/loss，eval_text 前 200 token）---"
python3 - <<'PY'
import json, urllib.request, time
text = open('/root/ascend-dev/eval_text.txt', encoding='utf-8').read()
rq = urllib.request.Request('http://127.0.0.1:8000/fnx/loss',
                            data=json.dumps({'text': text, 'max_tokens': 200}).encode(),
                            headers={'Content-Type': 'application/json'})
t0 = time.time()
d = json.load(urllib.request.urlopen(rq, timeout=1800))
print("  mean_nll=%.4f  ppl=%.3f  top1=%.1f%%  n=%d  (%.0fs)" % (
    d['mean_nll'], d['ppl'], d['top1_acc'] * 100, d['n'], time.time() - t0))
PY
echo "--- TPOT ---"
python3 - <<'PY'
import json, time, urllib.request
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
    return (tot - (first or 0)) / max(1, k - 1) * 1e3, k, first, tot
gen(16)
for n in (128, 128):
    t, k, first, tot = gen(n)
    print("  max_tokens=%d 出=%dtok TTFT=%.2fs 总=%.1fs TPOT=%.1f ms" % (n, k, first, tot, t))
PY
