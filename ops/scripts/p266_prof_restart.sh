#!/bin/bash
# ★ 重启 server 并打开 FNX_PROF=1，然后跑 3 个请求，dump /fnx/prof
#  目的：把每请求 ~5.7 s 的固定开销拆成 reset / prefill / decode / 采样 / HTTP
#  只用 curl + grep/sed，不依赖宿主 python
set -u
LOG=/root/ascend-dev/logs/server.log
echo "=== 重启：LPC=44 / NLAYERS=46 / PACKED_TORCH=1 / NO_PLE=1 / **FNX_PROF=1** ==="
docker exec fnx bash -lc "pkill -f 'python -m fnx.server' || true"; sleep 10
rm -f "$LOG"
T0=$(date +%s)
docker exec -d fnx bash -lc "cd /root/ascend-dev/fnx && FNX_PACK=0 FNX_EXP_G=64 FNX_PORT=8000 \
  FNX_LPC=44 FNX_NLAYERS=46 FNX_MOE_KERNEL=0 FNX_NO_PLE=1 FNX_PACKED_TORCH=1 FNX_PROF=1 \
  /root/ascend-dev/venv/bin/python -m fnx.server > $LOG 2>&1"
H=""
for i in $(seq 1 24); do
  sleep 10
  H=$(curl -s -m 4 http://127.0.0.1:8000/health)
  if [ -n "$H" ]; then
    echo "  ready @ $(date '+%T')   冷启动 = $(( $(date +%s) - T0 )) s"
    break
  fi
  grep -qiE "out of memory" "$LOG" 2>/dev/null && { echo "  !! OOM"; grep -i "Tried to allocate" "$LOG" | tail -1; break; }
done
[ -z "$H" ] && { echo "  !! 未就绪"; tail -5 "$LOG"; exit 1; }
grep -E "装配完成|图捕获" "$LOG" | tail -2 | sed 's/^/  /'

grab() { grep -o '"fnx_stats":{[^}]*}' | tail -1 | sed 's/^/   /'; }

echo
echo "=== R1: 非流式 你好 / max_tokens=8 ==="
T=$(date +%s%N)
curl -s -m 180 -X POST http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"你好"}],"max_tokens":8}' \
  | tee /tmp/r1.json | grab
echo "   curl 墙钟 = $(( ($(date +%s%N)-T)/1000000 )) ms"

echo "=== R2: 流式 你好 / max_tokens=8 ==="
T=$(date +%s%N)
curl -s -N -m 180 -X POST http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"你好"}],"max_tokens":8,"stream":true}' \
  | tee /tmp/r2.txt | grab
echo "   curl 墙钟 = $(( ($(date +%s%N)-T)/1000000 )) ms"

echo "=== R3: 流式 长 prompt / max_tokens=4 ==="
cat > /tmp/r3req.json <<'EOF'
{"messages":[{"role":"user","content":"请详细解释一下深度学习里的反向传播算法是怎么工作的，包括链式法则、梯度累加、以及为什么需要激活函数的导数，请尽量详细。"}],"max_tokens":4,"stream":true}
EOF
T=$(date +%s%N)
curl -s -N -m 300 -X POST http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' --data-binary @/tmp/r3req.json \
  | tee /tmp/r3.txt | grab
echo "   curl 墙钟 = $(( ($(date +%s%N)-T)/1000000 )) ms"

echo
echo "=== /fnx/prof ==="
curl -s -m 10 http://127.0.0.1:8000/fnx/prof > /tmp/prof.json
tr ',' '\n' < /tmp/prof.json | sed 's/^/  /'
