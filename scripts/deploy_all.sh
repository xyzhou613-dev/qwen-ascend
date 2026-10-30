#!/bin/bash
# =============================================================================
# 一键部署：自检 → 容器/环境 → 权重 → 起服务 → 验收
#
#   bash scripts/deploy_all.sh              # 全流程（缺 LQP 时会先建，约 860 s）
#   bash scripts/deploy_all.sh --skip-lqp   # 不建 LQP（服务退回运行时打包，冷启动 ~800 s）
#   bash scripts/deploy_all.sh --quick      # 验收跳过 24 题 MCQ
#
# 可用环境变量覆盖：RUN（默认 /root/ascend-dev）、CTN（fnx）、MDL（/models）、
#                  IMG（quay.io/ascend/vllm-ascend:v0.23.0）、PIP_INDEX_URL
# =============================================================================
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
SKIP_LQP=0; QUICK=""
for a in "$@"; do
  case "$a" in
    --skip-lqp) SKIP_LQP=1 ;;
    --quick)    QUICK="--quick" ;;
  esac
done
step() { printf '\n\033[36m──── %s ────\033[0m\n' "$*"; }

step "00 环境自检"
bash "$HERE/00_precheck.sh" || true     # 缺 LQP / venv 属于"可修复"，由后续步骤处理

step "10 容器与运行环境"
bash "$HERE/10_env.sh" || exit 1

step "20 权重"
if [ $SKIP_LQP -eq 1 ]; then bash "$HERE/20_weights.sh" skip || exit 1
else bash "$HERE/20_weights.sh" auto || exit 1; fi

step "30 部署服务"
bash "$HERE/30_deploy.sh" || exit 1

step "40 验收"
bash "$HERE/40_verify.sh" $QUICK
rc=$?

step "完成"
RUN=${RUN:-/root/ascend-dev}
echo "  接口：   http://127.0.0.1:8000/v1/chat/completions"
echo "  指标：   /metrics   /fnx/prof"
echo "  质量：   POST /fnx/loss  {\"text\": \"...\", \"max_tokens\": 200}"
echo "  重启：   systemctl restart fnx   或   bash $RUN/p275.sh"
echo "  日志：   $RUN/logs/server.log（上一份 server.log.prev）"
exit $rc
