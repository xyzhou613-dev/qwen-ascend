#!/bin/bash
# p335_fp.sh — 容器内运行：全层状态指纹对比
set -u
cd /root/ascend-dev/fnx || exit 1
FNX_LPC=45 FNX_NLAYERS=48 FNX_PACK=0 FNX_EXP_G=64 FNX_MOE_KERNEL=0 \
FNX_NO_PLE=1 FNX_PACKED_TORCH=1 FNX_DEQ_SMALL=1 FNX_DEQ_HEAD=1 FNX_DEQ_ATTN=0 \
FNX_LQP=/models/Qwen3.8-Flash-Next-LQP FNX_MOE_W4A16=1 \
FNX_W4A16_DIR=/models/Qwen3.8-Flash-Next-LQP-w4a16 FNX_DENSE_W8=1 \
FNX_DENSE_CAT=1 FNX_SMALL_MERGE=1 FNX_DN_RMS=1 FNX_CTX=2048 FNX_CHUNK=1 \
FNX_NO_CAPTURE=1 PYTHONPATH=. /root/ascend-dev/venv/bin/python tests/t_chunk_fp.py
echo "[p335 exit=$?]"
