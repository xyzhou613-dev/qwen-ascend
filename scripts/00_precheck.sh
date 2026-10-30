#!/bin/bash
# =============================================================================
# 00 · 环境自检（只读，不改任何东西）
#   检查：加速卡 / docker / 镜像 / 权重 / 磁盘 / eval_text
#   任何一项不满足都会逐条打印 FAIL，并给出修复建议。
# =============================================================================
set -u
RUN=${RUN:-/root/ascend-dev}          # 宿主侧工作目录（会被 bind 进容器）
CTN=${CTN:-fnx}                       # 容器名
IMG=${IMG:-quay.io/ascend/vllm-ascend:v0.23.0}
MDL=${MDL:-/models}
fail=0
ok()   { printf '  \033[32mOK\033[0m   %s\n' "$*"; }
bad()  { printf '  \033[31mFAIL\033[0m %s\n' "$*"; fail=$((fail+1)); }
info() { printf '  ..   %s\n' "$*"; }

echo "== 00 环境自检 =="

# ---- 加速卡 ----
NPU=$(npu-smi info 2>/dev/null | grep -cE '^\| [0-9]+ +910B4' || true)
if [ "${NPU:-0}" -ge 2 ]; then ok "检测到 $NPU 张 910B4"; else bad "只检测到 ${NPU:-0} 张 910B4（基线需要 2 张）"; fi
for d in /dev/davinci3 /dev/davinci5 /dev/davinci_manager /dev/devmm_svm /dev/hisi_hdc; do
  [ -e "$d" ] || bad "缺少设备 $d（容器要挂载它）"
done

# ---- docker ----
if command -v docker >/dev/null; then ok "docker $(docker --version | awk '{print $3}' | tr -d ,)"; else bad "没有 docker"; fi
if docker image inspect "$IMG" >/dev/null 2>&1; then
  ok "镜像已存在：$IMG"
else
  bad "镜像缺失：$IMG（先执行 docker pull，或参考 scripts/10_env.sh）"
fi

# ---- 容器 ----
if docker inspect "$CTN" >/dev/null 2>&1; then
  st=$(docker inspect "$CTN" --format '{{.State.Status}}')
  if [ "$st" = running ]; then ok "容器 $CTN 正在运行"; else bad "容器 $CTN 状态=$st（需要 running）"; fi
else
  info "容器 $CTN 不存在（scripts/10_env.sh 会创建）"
fi

# ---- 代码位置 ----
if [ -f "$RUN/fnx/fnx/server.py" ]; then ok "代码在 $RUN/fnx"; else bad "代码不在 $RUN/fnx（把仓库的 fnx/ tests/ tools/ 拷到 $RUN/fnx/）"; fi

# ---- 权重 ----
for d in Qwen3.8-Flash-Next Qwen3.8-Flash-Next-Q; do
  if [ -d "$MDL/$d" ]; then ok "权重目录 $MDL/$d（$(du -sh "$MDL/$d" 2>/dev/null | cut -f1)）"
  else bad "缺少 $MDL/$d"; fi
done
LQP=$MDL/Qwen3.8-Flash-Next-LQP
if [ -d "$LQP" ]; then
  n=$(ls "$LQP" 2>/dev/null | grep -c '^L[0-9].*safetensors')
  if [ "${n:-0}" -eq 48 ]; then ok "LQP artifact 完整（48 层）"; else bad "LQP 只有 ${n:-0}/48 层（用 scripts/20_weights.sh 重建）"; fi
else
  info "没有 LQP artifact：冷启动会退回运行时打包（约 800 s）。可用 scripts/20_weights.sh 生成"
fi
[ -f "$MDL/Qwen3.8-Flash-Next/config.json" ] && ok "原始 checkpoint（tokenizer/config）就位" || bad "缺少 $MDL/Qwen3.8-Flash-Next/config.json"

# ---- 评测文本 ----
if [ -f "$RUN/eval_text.txt" ]; then ok "eval_text.txt 就位（质量口径用）"; else bad "缺少 $RUN/eval_text.txt（mean_nll 需要它）"; fi

# ---- 磁盘 ----
avail=$(df -BG --output=avail "$MDL" 2>/dev/null | tail -1 | tr -dc '0-9')
if [ "${avail:-0}" -ge 60 ]; then ok "$MDL 剩余 ${avail}G"; else bad "$MDL 剩余仅 ${avail}G（建 LQP 需要 ~60G）"; fi

echo
if [ $fail -eq 0 ]; then echo "  00 自检通过"; exit 0; else echo "  00 自检发现 $fail 个问题（见上）"; exit 1; fi
