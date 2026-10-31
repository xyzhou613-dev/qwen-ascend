#!/bin/bash
# p344_moe_grp2.sh — 容器内运行：独占卡重测 MoE 分组（需先停生产）
set -u
cd /root/ascend-dev/fnx || exit 1
FNX_MOE_W4A16=1 FNX_W4A16_DIR=/models/Qwen3.8-Flash-Next-LQP-w4a16 \
FNX_QDIR=/models/Qwen3.8-Flash-Next-Q FNX_LQP=/models/Qwen3.8-Flash-Next-LQP \
FNX_DENSE_W8=1 FNX_EXP_G=64 FNX_PACKED_TORCH=1 \
PYTHONPATH=. /root/ascend-dev/venv/bin/python tests/t_moe_grp2.py
echo "[p344 exit=$?]"
