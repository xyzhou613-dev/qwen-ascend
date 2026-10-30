#!/bin/bash
# =============================================================================
# 20 · 权重准备（幂等）
#   有 48 层的 LQP artifact 就直接用；没有就用 tools/repack_v2.py 生成（约 860 s，可续跑）。
#   生成后必须过 tools/check_lqp.py（与运行时打包逐位等价，0.000e+00）。
#   也支持"跳过"：不建 LQP 时服务会退回运行时打包，冷启动约 800 s（功能等价，只是慢）。
# =============================================================================
set -u
RUN=${RUN:-/root/ascend-dev}
CTN=${CTN:-fnx}
MDL=${MDL:-/models}
LQP=${LQP:-$MDL/Qwen3.8-Flash-Next-LQP-g128}   # g128：scale 减半，省 19 MiB/层
PY="$RUN/venv/bin/python"
MODE=${1:-auto}      # auto | build | skip

echo "== 20 权重 =="
n=$(ls "$LQP" 2>/dev/null | grep -c '^L[0-9].*safetensors')
echo "  当前 LQP：${n:-0}/48 层"

if [ "${n:-0}" -eq 48 ] && [ "$MODE" != "build" ]; then
  echo "  LQP 完整，跳过生成"
  echo "  20 完成（用现成 artifact）"
  exit 0
fi
if [ "$MODE" = "skip" ]; then
  echo "  按要求跳过：服务将退回运行时打包（冷启动 ~800 s）"
  exit 0
fi

echo "  开始生成 LQP artifact（48 层，约 860 s，可续跑；输出 $LQP）"
docker exec "$CTN" bash -lc "cd $RUN/fnx && FNX_QDIR=$MDL/Qwen3.8-Flash-Next-Q \
    FNX_LQP=$LQP FNX_LQP_GS=128 $PY -u tools/repack_v2.py" || {
  echo "  ✗ 生成失败（可重跑本脚本续跑：已完成层不会重做）"; exit 1; }

echo "  等价性门禁：check_lqp.py（必须 0.000e+00）"
docker exec "$CTN" bash -lc "cd $RUN/fnx && FNX_QDIR=$MDL/Qwen3.8-Flash-Next-Q \
    FNX_LQP=$LQP $PY -u tools/check_lqp.py" || {
  echo "  ✗ artifact 与运行时打包不等价，不要上线"; exit 1; }

echo "  20 完成"
