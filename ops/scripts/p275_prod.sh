#!/bin/bash
# ★★ 生产启动脚本 v5（第30轮定稿 / 单实例安全版）
#   48/48 层：dev0 = 46 层图捕获（59.54 GiB）  dev1 = 2 层 eager + lm_head/mixer（3.19 GiB）
#   前缀缓存（思考关后 3 token）+ 对话缓存（LRU 4 槽）+ 停止符修正 + FNX_PROF=0
#   实测：mean_nll 2.1756 / TPOT 142.9 ms / 「你好」3.2 s（缓存命中 0.31 s）/ 冷启动 30 s
#
#   v4/v5 修的坑（都是第30轮实测踩出来的）：
#   1) 先杀看护再重启 —— 否则看护会在装配窗口内再触发一次重启，两边互相 pkill；
#   2) 重启后写 last_restart 时间戳，看护据此进入 150 s 宽限期（装配约 30 s）；
#   3) pkill 之后**要等进程真的退出**（NPU 显存不释放会直接 OOM），并强杀兜底；
#   4) 日志改成轮转，不再 rm —— 崩溃现场不能被自己删掉。
#
#   ⚠️ 计数坑：`pgrep -f 'python -m fnx.server'` 会同时匹配 `bash -lc` 包装进程，
#   所以「一个服务」会显示 count=2。判断"起没起来"请看 /health，不要看这个数。
#   ⚠️ 另一个坑：`pgrep ... || true` 会让 bash 保留 shell，pgrep 就匹配到自己 → 假残留，
#   所以命令行里的模式统一写成 `[p]ython`。
set -u
LOG=/root/ascend-dev/logs/server.log
STAMP=/root/ascend-dev/logs/last_restart
WD=${WD:-1}          # WD=1 时重启完顺带把看护拉起来

# 0) 先停看护（避免重启窗口内被二次触发）
pkill -f watchdog.sh 2>/dev/null || true
sleep 1

# 1) 停服务（必须杀干净：两个实例会抢 8000 端口与 NPU 显存）
docker exec fnx bash -lc "pkill -f '[p]ython -m fnx.server' || true"
for i in $(seq 1 30); do
  docker exec fnx bash -lc "pgrep -f '[p]ython -m fnx.server'" >/dev/null 2>&1 || break
  sleep 2
done
if docker exec fnx bash -lc "pgrep -f '[p]ython -m fnx.server'" >/dev/null 2>&1; then
  echo "  旧进程未退出 → 强杀"
  docker exec fnx bash -lc "pkill -9 -f '[p]ython -m fnx.server' || true"; sleep 5
fi

# 2) 日志轮转（保留上一份崩溃现场）+ 时间戳
[ -f "$LOG" ] && mv -f "$LOG" "$LOG.prev"
date +%s > "$STAMP"
T0=$(date +%s)

# 3) 起
docker exec -d fnx bash -lc "cd /root/ascend-dev/fnx && \
  FNX_PACK=0 FNX_EXP_G=64 FNX_MOE_KERNEL=0 FNX_NO_PLE=1 FNX_PACKED_TORCH=1 \
  FNX_LPC=46 FNX_NLAYERS=48 FNX_PORT=8000 \
  FNX_PREFIX_CACHE=1 FNX_CONV_CACHE=4 FNX_PROF=0 FNX_ENABLE_THINKING=0 \
  /root/ascend-dev/venv/bin/python -m fnx.server > $LOG 2>&1"

# 4) 等就绪
H=""
for i in $(seq 1 30); do
  sleep 10
  H=$(curl -s -m 4 http://127.0.0.1:8000/health || true)
  [ -n "$H" ] && { echo "ready 冷启动 = $(( $(date +%s) - T0 )) s  $H"; break; }
  if grep -qiE "out of memory|Tried to allocate" "$LOG" 2>/dev/null; then
    echo "!! OOM"; grep -iE "Tried to allocate" "$LOG" | tail -1; exit 1
  fi
done
[ -z "$H" ] && { echo "!! 未就绪"; tail -8 "$LOG"; exit 1; }
grep -E "装配 |图捕获 |前缀缓存|停止符|hbm[01]:" "$LOG" | sed 's/^/  /'

# 5) 看护（单一入口）
if [ "$WD" = 1 ]; then
  pkill -f watchdog.sh 2>/dev/null || true
  nohup bash /root/ascend-dev/watchdog.sh > /root/ascend-dev/logs/watchdog.log 2>&1 &
  sleep 1
  echo "  watchdog: $(pgrep -af watchdog.sh | head -1)"
fi
