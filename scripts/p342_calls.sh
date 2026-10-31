#!/bin/bash
# p342_calls.sh — 容器内运行：量 WQBM 每次调用的固定开销（需先停 fnx.service 独占 npu:0）
set -u
cd /root/ascend-dev/fnx || exit 1
FNX_MOE_W4A16=1 FNX_W4A16_DIR=/models/Qwen3.8-Flash-Next-LQP-w4a16 \
FNX_QDIR=/models/Qwen3.8-Flash-Next-Q FNX_LQP=/models/Qwen3.8-Flash-Next-LQP \
FNX_DENSE_W8=1 FNX_EXP_G=64 FNX_PACKED_TORCH=1 \
PYTHONPATH=. /root/ascend-dev/venv/bin/python tests/t_wqbm_calls.py
echo "[p342 exit=$?]"
