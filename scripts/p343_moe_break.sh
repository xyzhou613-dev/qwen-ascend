#!/bin/bash
# p343_moe_break.sh — 容器内运行：一层 MoE 逐算子拆解（需独占 npu:0）
set -u
cd /root/ascend-dev/fnx || exit 1
FNX_MOE_W4A16=1 FNX_W4A16_DIR=/models/Qwen3.8-Flash-Next-LQP-w4a16 \
FNX_QDIR=/models/Qwen3.8-Flash-Next-Q FNX_LQP=/models/Qwen3.8-Flash-Next-LQP \
FNX_DENSE_W8=1 FNX_EXP_G=64 FNX_PACKED_TORCH=1 \
PYTHONPATH=. /root/ascend-dev/venv/bin/python tests/t_moe_break.py
echo "[p343 exit=$?]"
