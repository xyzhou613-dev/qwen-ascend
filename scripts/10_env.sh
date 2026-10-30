#!/bin/bash
# =============================================================================
# 10 · 准备容器与运行环境（幂等，可重复执行）
#   - 容器不存在就按基线的设备/挂载/网络创建；已存在就检查关键配置
#   - 虚拟环境 /root/ascend-dev/venv 缺失或导入失败时，在容器内创建并安装依赖
# =============================================================================
set -u
RUN=${RUN:-/root/ascend-dev}
CTN=${CTN:-fnx}
IMG=${IMG:-quay.io/ascend/vllm-ascend:v0.23.0}
MDL=${MDL:-/models}
PY="$RUN/venv/bin/python"
REPO=${REPO:-$RUN/fnx}                 # 代码目录（仓库里的 fnx/tests/tools 拷到这里）
[ -n "${PIP_INDEX_URL:-}" ] && PIPARG="-i $PIP_INDEX_URL" || PIPARG=""
[ -n "${PIP_TRUSTED_HOST:-}" ] && PIPARG="$PIPARG --trusted-host $PIP_TRUSTED_HOST" || true

echo "== 10 容器与运行环境 =="

# ---------- 1. 容器 ----------
if docker inspect "$CTN" >/dev/null 2>&1; then
  echo "  容器 $CTN 已存在（$(docker inspect "$CTN" --format '{{.State.Status}}')）"
  docker inspect "$CTN" --format '{{json .HostConfig.Binds}}' | grep -q "$RUN" \
    || echo "  ⚠️ 容器没有挂载 $RUN，运行脚本会看不到代码"
else
  echo "  创建容器 $CTN（沿用镜像 $IMG）"
  docker image inspect "$IMG" >/dev/null 2>&1 || { echo "  镜像缺失，先 docker pull $IMG"; exit 1; }
  docker run -d --name "$CTN" --restart unless-stopped \
    -v "$MDL:$MDL" -v "$RUN:$RUN" \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
    -v /usr/local/dcmi:/usr/local/dcmi:ro \
    -v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi:ro \
    -v /etc/ascend_install.info:/etc/ascend_install.info:ro \
    --device /dev/davinci3 --device /dev/davinci5 \
    --device /dev/davinci_manager --device /dev/devmm_svm --device /dev/hisi_hdc \
    --network host --ipc host \
    -e TORCH_DEVICE_BACKEND_AUTOLOAD=1 \
    "$IMG" bash -c 'sleep 100000000'
fi

# ---------- 2. 代码就位 ----------
mkdir -p "$RUN/fnx"
for sub in fnx tests tools; do
  if [ -d "$(dirname "$0")/../$sub" ]; then
    cp -r "$(dirname "$0")/../$sub" "$RUN/fnx/" 2>/dev/null || true
  fi
done
if [ -f "$RUN/fnx/fnx/server.py" ]; then echo "  代码已就位：$RUN/fnx"
else echo "  ✗ 代码未就位（把仓库的 fnx/ tests/ tools/ 放到 $RUN/fnx/）"; exit 1; fi

# ---------- 3. venv ----------
# ⚠️ 不能在宿主侧用 `-x $PY` 判断：venv 里的 python 是指向**容器内**解释器的符号链接，
#    宿主上看是不可执行的，会误判成"没装"并重跑 pip（实测踩过）。
have=0
if docker exec "$CTN" test -x "$PY" 2>/dev/null; then
  if docker exec "$CTN" bash -lc "$PY -c 'import torch, torch_npu, transformers, fastapi, uvicorn, safetensors'" >/dev/null 2>&1; then
    have=1
  fi
fi

if [ $have -eq 0 ]; then
  echo "  在容器内准备 venv：$RUN/venv"
  docker exec "$CTN" bash -lc "
    set -e
    [ -x $PY ] || python3 -m venv $RUN/venv
    $PY -m pip install -q --upgrade pip $PIPARG
    # torch / torch_npu 优先用镜像自带的（避免把加速版覆盖成普通版）
    if ! $PY -c 'import torch, torch_npu' 2>/dev/null; then
      echo '    torch/torch_npu 不可用，尝试安装（可能很慢）'
      $PY -m pip install -q torch==2.10.0 torch_npu==2.10.0.post4 $PIPARG
    fi
    $PY -m pip install -q numpy==1.26.4 transformers==5.21.0 safetensors==0.8.0 \
        fastapi==0.123.10 uvicorn==0.52.3 $PIPARG
  " || { echo "  ✗ venv 创建失败（检查网络 / PIP_INDEX_URL）"; exit 1; }
else
  echo "  venv 已就绪，跳过安装"
fi

docker exec "$CTN" "$PY" - <<'PY' || { echo "  ✗ 依赖导入失败"; exit 1; }
import torch, torch_npu, transformers, fastapi, uvicorn, safetensors, numpy
print(f"  torch={torch.__version__} torch_npu={torch_npu.__version__} "
      f"transformers={transformers.__version__} fastapi={fastapi.__version__}")
PY

# ---------- 4. 评测文本 ----------
if [ ! -f "$RUN/eval_text.txt" ] && [ -f "$(dirname "$0")/../tests/eval_text.txt" ]; then
  cp "$(dirname "$0")/../tests/eval_text.txt" "$RUN/eval_text.txt"
fi
[ -f "$RUN/eval_text.txt" ] && echo "  eval_text.txt 就位" || echo "  ⚠️ 缺 $RUN/eval_text.txt（/fnx/loss 需要它）"

echo "  10 完成"
