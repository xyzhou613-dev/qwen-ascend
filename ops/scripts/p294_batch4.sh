#!/bin/bash
# ★ B=4 测量：等价性 + 每步成本（用于给出吞吐投影）
set -u
B=${1:-4}
echo "== 停服务，以 FNX_BATCH=$B 启动 =="
systemctl stop fnx.service
pkill -f watchdog.sh 2>/dev/null || true
docker exec fnx bash -lc "pkill -f '[p]ython -m fnx.server' || true"
for i in $(seq 1 30); do
  docker exec fnx bash -lc "pgrep -f '[p]ython -m fnx.server'" >/dev/null 2>&1 || break
  sleep 2
done
rm -f /root/ascend-dev/logs/batch.log
docker exec -d fnx bash -lc "cd /root/ascend-dev/fnx && \
  FNX_PACK=0 FNX_EXP_G=64 FNX_MOE_KERNEL=0 FNX_NO_PLE=1 FNX_PACKED_TORCH=1 \
  FNX_LPC=46 FNX_NLAYERS=48 FNX_PORT=8000 FNX_BATCH=$B \
  FNX_PREFIX_CACHE=1 FNX_CONV_CACHE=4 FNX_PROF=0 FNX_ENABLE_THINKING=0 \
  /root/ascend-dev/venv/bin/python -m fnx.server > /root/ascend-dev/logs/batch.log 2>&1"
H=""
for i in $(seq 1 30); do
  sleep 10
  H=$(curl -s -m 4 http://127.0.0.1:8000/health || true)
  [ -n "$H" ] && { echo "  ready"; break; }
  grep -qiE "out of memory|Tried to allocate|Traceback" /root/ascend-dev/logs/batch.log 2>/dev/null && {
    echo "  !! 失败"; grep -vE "^\[W[0-9]" /root/ascend-dev/logs/batch.log | tail -12 | sed 's/^/    /'; exit 1; }
done
[ -z "$H" ] && { echo "  !! 未就绪"; exit 1; }
grep -E "hbm[01]:" /root/ascend-dev/logs/batch.log | sed 's/^/  /'

echo
echo "== 等价性 /fnx/loss（B=1 基线 2.1756）=="
python3 - <<'PY'
import json, urllib.request
text = open('/root/ascend-dev/eval_text.txt', encoding='utf-8').read()
r = urllib.request.Request('http://127.0.0.1:8000/fnx/loss',
    data=json.dumps({"text": text, "max_tokens": 200}).encode(),
    headers={'Content-Type': 'application/json'})
d = json.load(urllib.request.urlopen(r, timeout=900))
print(f"  mean_nll={d['mean_nll']:.4f} ppl={d['ppl']:.2f} top1={d['top1_acc']*100:.2f}%")
PY

echo
echo "== 单请求每步成本（B=$B 时是填充态）=="
python3 - <<'PY'
import json, time, urllib.request
URL = "http://127.0.0.1:8000/v1/chat/completions"
def gen(mx):
    body = json.dumps({"messages": [{"role": "user", "content": "请写一段约200字的散文"}],
                       "max_tokens": mx, "temperature": 0, "stream": True}).encode()
    r = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    t0 = time.perf_counter(); first = None; k = 0
    for line in urllib.request.urlopen(r, timeout=900):
        s = line.decode("utf-8", "ignore").strip()
        if not s.startswith("data: "): continue
        p = s[6:]
        if p == "[DONE]": break
        d = json.loads(p)
        if d["choices"][0]["delta"].get("content"):
            k += 1
            if first is None: first = time.perf_counter() - t0
    tot = time.perf_counter() - t0
    print(f"  出={k:3d}tok TTFT={first:5.2f}s 总={tot:5.2f}s "
          f"每步={(tot-first)/max(1,k-1)*1e3:6.1f} ms")
gen(64)
gen(64)
PY
