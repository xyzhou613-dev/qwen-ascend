#!/bin/bash
# r100: validated 256k profile, exact caches and native sparse prefill.
# BF16 KV, FP32 recurrent state, double-buffered chunk prefill.
set -u
# r52 context profile
source /root/ascend-dev/r52/active_profile.env
LOG=/root/ascend-dev/logs/server.log
STAMP=/root/ascend-dev/logs/last_restart
WD=${WD:-1}
W4A16=${W4A16:-1}
pkill -f watchdog.sh 2>/dev/null || true
sleep 1
# ★ systemctl stop 不会杀容器里的进程（systemd cgroup 不含 docker exec 的子进程），必须显式杀
docker exec fnx bash -lc "pkill -f '[p]ython -m fnx.server' || true"
for i in $(seq 1 60); do
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
  ASCEND_RT_VISIBLE_DEVICES=0 FNX_PIPELINE=1 FNX_PIPELINE_DEVICE1=1 FNX_PIPELINE_SPLIT=$LPC FNX_PACK=0 FNX_EXP_G=64 FNX_MOE_KERNEL=0 FNX_NO_PLE=1 FNX_PACKED_TORCH=1 \
  FNX_LPC=$LPC FNX_NLAYERS=48 FNX_PORT=8000 FNX_BATCH=1 \
  FNX_PREFIX_CACHE=1 FNX_CONV_CACHE=1 FNX_CACHE_GPU=1 FNX_PROF=0 FNX_ENABLE_THINKING=0 \
  FNX_DEQ_SMALL=1 FNX_DEQ_HEAD=1 FNX_DEQ_ATTN=48 \
  FNX_LQP=/models/Qwen3.8-Flash-Next-LQP \
  FNX_MOE_W4A16=$W4A16 \
  FNX_W4A16_DIR=/models/Qwen3.8-Flash-Next-LQP-w4a16 \
  FNX_DENSE_W8=1 FNX_DENSE_CAT=1 FNX_SMALL_MERGE=1 FNX_DN_RMS=1 FNX_CTX=$CTX FNX_CHUNK=8192 FNX_CHUNK_SIZES=8192,4096,512,128 FNX_CHUNK_POOL=shared FNX_CHUNK_GRAPH=1 FNX_INDEXER=1 FNX_QSA_SPARSE=1 FNX_DELTA_BATCH=1 FNX_INDEX_CACHE=1 FNX_INDEX_TILE=512 FNX_QSA_TILE=128 FNX_KV_POSITION_MAJOR=1 FNX_SPARSE_FLASH=1 FNX_DN_DECODE_BMM=0 FNX_GRAPH_G1=0 FNX_GRAPH_HEAD=0 FNX_MOE_GROUP=1 FNX_MOE_GROUP_SPLIT=0 FNX_QSA_GQA=1 FNX_PREFILL_DN=fp32_chunk FNX_DELTA_SOLVER=blockmatmul \
  FNX_MOE_FAST=1 FNX_MOE_ROUTER_FAST=0 FNX_MOE_ACT_BF16=0 \
  FNX_PIPELINE_FUSED_HEAD=1 \
  FNX_W4_SCALE_CACHE=1 \
  FNX_DECODE_DN_GMM=1 \
  FNX_MOE_SWIGLU_FP32=1 \
  FNX_INDEX_BUCKETS=1 \
  FNX_NATIVE_SPARSE=1 \
  FNX_NATIVE_DUAL=1 \
  FNX_ROPE_CACHE=1 \
  FNX_DELTA_WINDOW=2048 \
  FNX_MIX_PTO=0 \
  FNX_DN_DECODE_PTO=0 \
  FNX_INDEX_CAPS=16384,32768,65536,98304,131072,196608,262144 \
  FNX_PIPELINE_TRANSPORT=hybrid \
  FNX_INDEX_UNIQUE=1 FNX_STATE_DUAL_SLIM=1 \
  FNX_MOE_INVERSE=1 \
  /root/ascend-dev/venv/bin/python -m fnx.server > $LOG 2>&1"
H=""
for i in $(seq 1 60); do
  sleep 10
  H=$(curl -fs -m 4 http://127.0.0.1:8000/health || true)
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