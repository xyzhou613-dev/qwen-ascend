#!/bin/bash
# p341_moe_grp.sh — 容器内运行：MoE 三条路径对照（dev1，不动生产 dev0）
set -u
cd /root/ascend-dev/fnx || exit 1
FNX_MOE_W4A16=1 FNX_W4A16_DIR=/models/Qwen3.8-Flash-Next-LQP-w4a16 \
FNX_QDIR=/models/Qwen3.8-Flash-Next-Q FNX_LQP=/models/Qwen3.8-Flash-Next-LQP \
FNX_DENSE_W8=1 FNX_EXP_G=64 FNX_PACKED_TORCH=1 \
PYTHONPATH=. /root/ascend-dev/venv/bin/python tests/t_moe_grp.py
echo "[p341 exit=$?]"
