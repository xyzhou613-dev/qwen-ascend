#!/bin/bash
# start_vllm_27b.sh —— 在容器内用 vLLM-Ascend 起 Qwen3.8-27B（bf16，TP=2）
set -u
cd /models || exit 1
mkdir -p /root/ascend-dev/logs
LOG=/root/ascend-dev/logs/vllm27b.log
export ASCEND_RT_VISIBLE_DEVICES=0,1
exec vllm serve /models/Qwen3.8-27B \
  --served-model-name qwen3.8-27b \
  --tensor-parallel-size 2 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.90 \
  --host 0.0.0.0 --port 8000 \
  --trust-remote-code \
  --limit-mm-per-prompt '{"image": 0, "video": 0}' \
  > "$LOG" 2>&1
