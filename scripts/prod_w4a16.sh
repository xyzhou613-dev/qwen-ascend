#!/bin/bash
# 生产启动脚本 —— 基线 baseline-r48（48 层：dev0 45 图捕获 + dev1 3 eager + head）
# ★ 2026-10-31 修正：本文件在 10:40 被改成实验配置（FNX_LQP=...-LQP-g128 + FNX_DEQ_ATTN=10），
#   导致 TPOT 64 -> 74 ms、dev0 58.97 -> 59.53 GiB（DEQ_ATTN 恰好多占 0.55 GiB）。
#   现恢复为 r47/r48 的生产口径：plain LQP + DEQ_ATTN=0。
set -u
LOG=/root/ascend-dev/logs/server.log
STAMP=/root/ascend-dev/logs/last_restart
WD=${WD:-1}
W4A16=${W4A16:-1}
pkill -f watchdog.sh 2>/dev/null || true
sleep 1
# ★ systemctl stop 不会杀容器里的进程（systemd cgroup 不含 docker exec 的子进程），必须显式杀
docker exec fnx bash -lc "pkill -f '[p]ython -m fnx.server' || true"
for i in $(seq 1 30); do
  docker exec fnx bash -lc "pgrep -f '[p]ython -m fnx.server'" >/dev/null 2>&1 || break
  sleep 2
done
if docker exec fnx bash -lc "pgrep -f '[p]ython -m fnx.server'" >/dev/null 2>&1; then
  docker exec fnx bash -lc "pkill -9 -f '[p]ython -m fnx.server' || true"; sleep 5
fi
[ -f "$LOG" ] && mv -f "$LOG" "$LOG.prev"
date +%s > "$STAMP"
T0=$(date +%s)
docker exec -d fnx bash -lc "cd /root/ascend-dev/fnx && \
  FNX_PACK=0 FNX_EXP_G=64 FNX_MOE_KERNEL=0 FNX_NO_PLE=1 FNX_PACKED_TORCH=1 \
  FNX_LPC=45 FNX_NLAYERS=48 FNX_PORT=8000 FNX_BATCH=1 \
  FNX_PREFIX_CACHE=1 FNX_CONV_CACHE=4 FNX_PROF=0 FNX_ENABLE_THINKING=0 \
  FNX_DEQ_SMALL=1 FNX_DEQ_HEAD=1 FNX_DEQ_ATTN=0 \
  FNX_LQP=/models/Qwen3.8-Flash-Next-LQP \
  FNX_MOE_W4A16=$W4A16 \
  FNX_W4A16_DIR=/models/Qwen3.8-Flash-Next-LQP-w4a16 \
  FNX_DENSE_W8=1 FNX_DENSE_CAT=1 FNX_SMALL_MERGE=1 FNX_DN_RMS=1 FNX_CTX=2048 \
  /root/ascend-dev/venv/bin/python -m fnx.server > $LOG 2>&1"
H=""
for i in $(seq 1 30); do
  sleep 10
  H=$(curl -s -m 4 http://127.0.0.1:8000/health || true)
  [ -n "$H" ] && { echo "ready 冷启动 = $(( $(date +%s) - T0 )) s  $H"; break; }
  grep -qiE "out of memory|Tried to allocate" "$LOG" 2>/dev/null && {
    echo "!! OOM"; grep -iE "Tried to allocate" "$LOG" | tail -1; exit 1; }
done
[ -z "$H" ] && { echo "!! 未就绪"; tail -8 "$LOG"; exit 1; }
grep -E "装配 |图捕获 |停止符|前缀缓存|hbm[01]:" "$LOG" | sed 's/^/  /'
if [ "$WD" = 1 ]; then
  pkill -f watchdog.sh 2>/dev/null || true
  nohup bash /root/ascend-dev/watchdog.sh > /root/ascend-dev/logs/watchdog.log 2>&1 &
  sleep 1
  echo "  watchdog: $(pgrep -af watchdog.sh | head -1)"
fi