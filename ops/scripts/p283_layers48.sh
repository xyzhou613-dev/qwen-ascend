#!/bin/bash
# ★ 48 层 vs 46 层：同一 /fnx/loss 口径对比（第 30 轮：最后 2 层是否值得搬上 dev1）
set -u
LOG=/root/ascend-dev/logs/server.log
NL=${1:-48}
LPC=${2:-44}
E="FNX_PACK=0 FNX_EXP_G=64 FNX_MOE_KERNEL=0 FNX_NO_PLE=1 FNX_PACKED_TORCH=1"
echo "=== 重启 NLAYERS=$NL LPC=$LPC（44 层图捕获 + $((NL-LPC)) 层 eager 在 dev1）==="
docker exec fnx bash -lc "pkill -f 'python -m fnx.server' || true"; sleep 12
rm -f "$LOG"
T0=$(date +%s)
docker exec -d fnx bash -lc "cd /root/ascend-dev/fnx && $E FNX_LPC=$LPC FNX_NLAYERS=$NL \
  FNX_PORT=8000 FNX_PREFIX_CACHE=1 FNX_CONV_CACHE=2 FNX_PROF=0 FNX_ENABLE_THINKING=0 \
  /root/ascend-dev/venv/bin/python -m fnx.server > $LOG 2>&1"
H=""
for i in $(seq 1 24); do
  sleep 10
  H=$(curl -s -m 4 http://127.0.0.1:8000/health)
  [ -n "$H" ] && { echo "  ready 冷启动 = $(( $(date +%s) - T0 )) s"; break; }
  grep -qiE "out of memory|Tried to allocate" "$LOG" 2>/dev/null && { echo "  !! OOM"; grep -iE "Tried to allocate|out of memory" "$LOG" | tail -2; exit 1; }
done
[ -z "$H" ] && { echo "  !! 未就绪"; tail -8 "$LOG"; exit 1; }
grep -E "装配完成|装配 |图捕获 |前缀缓存|hbm[01]:" "$LOG" | sed 's/^/  /'

echo
echo "=== 质量口径 /fnx/loss（46 层基线 3.1874 / 48.50% / rank 1）==="
python3 - <<'PY'
import json, urllib.request
text = open('/root/ascend-dev/eval_text.txt', encoding='utf-8').read()
r = urllib.request.Request('http://127.0.0.1:8000/fnx/loss',
    data=json.dumps({"text": text, "max_tokens": 200}).encode(),
    headers={'Content-Type': 'application/json'})
d = json.load(urllib.request.urlopen(r, timeout=900))
print(f"  n={d['n']} mean_nll={d['mean_nll']:.4f} ppl={d['ppl']:.2f} "
      f"top1={d['top1_acc']*100:.2f}% median_rank={d['median_rank']}")
print(f"  argmax_minus_tgt_top5={d['argmax_minus_tgt_top5']}")
PY

echo
echo "=== 稳态 TPOT（长输出）+ 'Qwen' 复现 ==="
python3 - <<'PY'
import json, time, urllib.request
URL = "http://127.0.0.1:8000/v1/chat/completions"
def gen(msg, n, tag=""):
    body = json.dumps({"messages": [{"role": "user", "content": msg}],
                       "max_tokens": n, "temperature": 0, "stream": True}).encode()
    r = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    t0 = time.perf_counter(); first = None; k = 0; txt = []
    for line in urllib.request.urlopen(r, timeout=900):
        s = line.decode("utf-8", "ignore").strip()
        if not s.startswith("data: "): continue
        p = s[6:]
        if p == "[DONE]": break
        d = json.loads(p)
        c = d["choices"][0]["delta"].get("content")
        if c:
            k += 1; txt.append(c)
            if first is None: first = time.perf_counter() - t0
    tot = time.perf_counter() - t0
    print(f"  {tag or msg[:14]!r:18s} 出={k:3d}tok TTFT={first:6.2f}s 总={tot:6.2f}s "
          f"TPOT={(tot-first)/max(1,k-1)*1e3:6.1f}ms")
    return "".join(txt)
gen("请写一段关于秋天的散文，约200字", 200, "长输出")
a = gen("你好，请用一句话介绍你自己", 96)
print(f"  A: {a[:160]!r}")
b = gen("用一句话说明 MoE 和 Qwen 分别是什么", 200)
print(f"  A: {b[:200]!r}")
PY
