#!/bin/bash
# p332: 用指定 FNX_CTX 启动服务并报告是否成功 + 显存
#   用法：CTX=16384 bash p332_ctx.sh
set -u
CTX=${CTX:-2048}
CTN=${CTN:-fnx}
PY=/root/ascend-dev/venv/bin/python
LOG=/root/ascend-dev/logs/server.log
RUN=/root/ascend-dev

systemctl stop fnx.service >/dev/null 2>&1 || true
pkill -f watchdog.sh 2>/dev/null || true
docker exec $CTN bash -lc "pkill -9 -f '[p]ython -m fnx.server' || true"
sleep 4
[ -f "$LOG" ] && mv -f "$LOG" "$LOG.prev"
docker exec -d $CTN bash -lc "cd $RUN/fnx && \
  FNX_PACK=0 FNX_EXP_G=64 FNX_MOE_KERNEL=0 FNX_NO_PLE=1 FNX_PACKED_TORCH=1 \
  FNX_LPC=45 FNX_NLAYERS=48 FNX_PORT=8000 FNX_BATCH=1 FNX_CTX=$CTX \
  FNX_PREFIX_CACHE=1 FNX_CONV_CACHE=4 FNX_PROF=0 FNX_ENABLE_THINKING=0 \
  FNX_DEQ_SMALL=1 FNX_DEQ_HEAD=1 FNX_DEQ_ATTN=0 \
  FNX_LQP=/models/Qwen3.8-Flash-Next-LQP \
  FNX_MOE_W4A16=1 FNX_W4A16_DIR=/models/Qwen3.8-Flash-Next-LQP-w4a16 \
  FNX_DENSE_W8=1 FNX_DENSE_CAT=1 FNX_SMALL_MERGE=1 FNX_DN_RMS=1 \
  $PY -m fnx.server > $LOG 2>&1"
T0=$(date +%s)
for i in $(seq 1 45); do
  sleep 8
  H=$(curl -s -m 4 http://127.0.0.1:8000/health || true)
  [ -n "$H" ] && { echo "CTX=$CTX ready 冷启动=$(( $(date +%s) - T0 ))s $H"; break; }
  grep -qiE "Tried to allocate|out of memory|Traceback" "$LOG" 2>/dev/null && {
    echo "CTX=$CTX FAIL"; grep -iE "Tried to allocate|out of memory|Error" "$LOG" | tail -3; exit 1; }
done
grep -E "hbm0:|图捕获完成|前缀缓存" "$LOG" | tail -3
