#!/bin/bash
# p330: r37 配置下的重新采集（msProf 算子分布 + 模块拆分）
set -u
export FNX_PACK=0 FNX_EXP_G=64 FNX_MOE_KERNEL=0 FNX_NO_PLE=1 FNX_PACKED_TORCH=1
export FNX_LPC=45 FNX_NLAYERS=48 FNX_BATCH=1 FNX_PROF=0 FNX_ENABLE_THINKING=0
export FNX_DEQ_SMALL=1 FNX_DEQ_HEAD=1 FNX_DEQ_ATTN=0
export FNX_DENSE_W8=1
export FNX_LQP=/models/Qwen3.8-Flash-Next-LQP
export FNX_MOE_W4A16=1
export FNX_W4A16_DIR=/models/Qwen3.8-Flash-Next-LQP-w4a16
PY=/root/ascend-dev/venv/bin/python
cd /root/ascend-dev/fnx || exit 1
rm -rf /tmp/prof_fnx
echo "=========== 1) msProf（captured，r37）==========="
$PY -u tests/t_msprof.py 2>&1 | grep -vE "Warning|warn" | tail -45
echo
echo "=========== 2) 模块拆分（eager + monkeypatch）==========="
PR_LPC=45 PR_NL=48 PR_DL=12 $PY -u tests/t_split_r37.py 2>&1 | grep -vE "Warning|warn" | tail -30
