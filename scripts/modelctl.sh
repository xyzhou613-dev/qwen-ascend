#!/bin/bash
# modelctl.sh —— 按配置启停在 910B4 上的不同模型（同一时刻只跑一个）
#
#   modelctl.sh list                列出配置里可启动的模型
#   modelctl.sh status              看现在跑的是哪个、端口/模型名/显存
#   modelctl.sh start <模型>        停掉其它模型 → 等显存释放 → 启动 → 等健康检查通过
#   modelctl.sh stop [模型]         停掉指定模型（不给参数=停全部）
#   modelctl.sh restart <模型>      等价于 start（start 会先停全部）
#   modelctl.sh logs <模型> [行数]   看该模型本次启动日志
#
# 配置：/root/ascend-dev/models.json（可用环境变量 MODELCTL_CONF 覆盖）
# 依赖：docker 容器 fnx、systemd 单元 fnx.service、curl、npu-smi、python3
set -u

CONF=${MODELCTL_CONF:-/root/ascend-dev/models.json}
CONTAINER=${MODELCTL_CONTAINER:-fnx}
VLLM_PATTERN='vllm serve'
FNX_PATTERN='[p]ython -m fnx.server'

die()  { echo "错误: $*" >&2; exit 1; }
info() { echo "[$(date '+%H:%M:%S')] $*"; }

[ -f "$CONF" ] || die "找不到配置文件 $CONF"
command -v python3 >/dev/null || die "需要 python3"

# ---------- 配置解析 ----------
load_profile() {
    python3 - "$CONF" "$1" <<'PY'
import json, shlex, sys
conf, name = sys.argv[1], sys.argv[2]
cfg = json.load(open(conf, encoding="utf-8"))
models = cfg.get("models") or {}
if name not in models:
    sys.exit(3)
m = dict(cfg.get("defaults") or {})
m.update(models[name])
m.setdefault("port", cfg.get("default_port", 8000))
m.setdefault("model_id", name)
m.setdefault("kind", "vllm")
for key, val in (
    ("KIND", m.get("kind")),
    ("DESC", m.get("desc", "")),
    ("MODEL_ID", m.get("model_id")),
    ("MODEL_PATH", m.get("model_path", "")),
    ("PORT", str(m.get("port"))),
    ("TP", str(m.get("tp", 1))),
    ("MAX_LEN", str(m.get("max_model_len", 32768))),
    ("GPU_UTIL", str(m.get("gpu_memory_utilization", 0.9))),
    ("DEVICES", m.get("devices", "0,1")),
    ("VLLM_ARGS", m.get("vllm_args", "")),
    ("SYSTEMD", m.get("systemd", "")),
    ("HEALTH", m.get("health", "/health")),
    ("START_LOG", m.get("start_log", "")),
    ("HEALTH_T", str(m.get("health_timeout_s", 900))),
    ("STOP_T", str(m.get("stop_timeout_s", 90))),
):
    print("%s=%s" % (key, shlex.quote(str(val))))
PY
}

list_models() {
    python3 - "$CONF" <<'PY'
import json, sys
cfg = json.load(open(sys.argv[1], encoding="utf-8"))
for name, m in (cfg.get("models") or {}).items():
    print("  %-12s %-5s %-28s %s" % (name, m.get("kind", "vllm"),
                                     m.get("model_id") or name, m.get("desc", "")))
PY
}

list_systemd_units() {
    python3 - "$CONF" <<'PY'
import json, sys
cfg = json.load(open(sys.argv[1], encoding="utf-8"))
seen = []
for m in (cfg.get("models") or {}).values():
    u = m.get("systemd")
    if u and u not in seen:
        seen.append(u)
for u in seen:
    print(u)
PY
}

# ---------- 状态探测 ----------
vllm_running() { docker exec "$CONTAINER" bash -lc "pgrep -f '$VLLM_PATTERN'" >/dev/null 2>&1; }
fnx_running()  { docker exec "$CONTAINER" bash -lc "pgrep -f '$FNX_PATTERN'" >/dev/null 2>&1; }

npu_busy() {
    npu-smi info 2>/dev/null | grep -qE "VLLMWorker|fnx.server|[0-9]+ +python" && return 0
    return 1
}

wait_npu_free() {
    local t=${1:-90} i=0
    while npu_busy; do
        i=$((i + 2)); [ "$i" -ge "$t" ] && return 1; sleep 2
    done
    return 0
}

wait_health() {  # port path limit
    local port=$1 path=$2 limit=${3:-900} i=0 code
    while :; do
        code=$(curl -s -o /dev/null -m 4 -w '%{http_code}' "http://127.0.0.1:${port}${path}" || true)
        [ "$code" = "200" ] && return 0
        i=$((i + 5)); [ "$i" -ge "$limit" ] && return 1
        sleep 5
    done
}

# ---------- 停止 ----------
stop_vllm() {
    info "停止 vLLM"
    docker exec "$CONTAINER" bash -lc "pkill -f '$VLLM_PATTERN'" >/dev/null 2>&1 || true
    sleep 3
    docker exec "$CONTAINER" bash -lc "pkill -9 -f '$VLLM_PATTERN'" >/dev/null 2>&1 || true
}

stop_fnx() {
    info "停止 fnx（systemd + 容器内进程）"
    local unit
    for unit in $(list_systemd_units); do
        systemctl stop "$unit" >/dev/null 2>&1 || true
    done
    pkill -f watchdog.sh >/dev/null 2>&1 || true
    # ★ systemctl stop 不会杀 docker exec 起的子进程，必须显式杀，否则显存不释放
    docker exec "$CONTAINER" bash -lc "pkill -9 -f '$FNX_PATTERN'" >/dev/null 2>&1 || true
}

stop_all() {  # $1=超时秒数（可选）
    local t=${1:-90}
    fnx_running  && stop_fnx
    vllm_running && stop_vllm
    # 兜底：两种都再清一遍
    for unit in $(list_systemd_units); do systemctl stop "$unit" >/dev/null 2>&1 || true; done
    docker exec "$CONTAINER" bash -lc \
        "pkill -9 -f '$FNX_PATTERN'; pkill -9 -f '$VLLM_PATTERN'" >/dev/null 2>&1 || true
    if ! wait_npu_free "$t"; then
        die "显存未在 ${t}s 内释放，请手动检查 npu-smi info"
    fi
}

# ---------- 启动 ----------
start_model() {
    local name=$1
    eval "$(load_profile "$name")" || die "配置里没有模型 '$name'（用 list 查看）"

    info "目标: $name  kind=$KIND  model_id=$MODEL_ID  port=$PORT"
    stop_all "${STOP_T:-90}"
    info "显存已释放"

    [ -n "${START_LOG:-}" ] && mkdir -p "$(dirname "$START_LOG")"

    case "$KIND" in
    fnx)
        [ -n "${SYSTEMD:-}" ] || die "fnx 类型必须配置 systemd"
        info "启动 $SYSTEMD（冷启动约 2–4 min：装配 + 前缀缓存 + 图捕获）"
        systemctl start "$SYSTEMD" || die "systemctl start $SYSTEMD 失败"
        ;;
    vllm)
        [ -n "$MODEL_PATH" ] || die "vllm 类型必须配置 model_path"
        [ -d "$MODEL_PATH" ] || die "权重目录不存在: $MODEL_PATH"
        local log=${START_LOG:-/root/ascend-dev/logs/modelctl-${name}.log}
        info "启动 vLLM（TP=$TP, max_model_len=$MAX_LEN；首次需图编译，约 2–6 min）"
        docker exec -d "$CONTAINER" bash -lc \
            "cd /models && ASCEND_RT_VISIBLE_DEVICES='$DEVICES' \
             vllm serve '$MODEL_PATH' --served-model-name '$MODEL_ID' \
             --tensor-parallel-size $TP --max-model-len $MAX_LEN \
             --gpu-memory-utilization $GPU_UTIL --host 0.0.0.0 --port $PORT \
             $VLLM_ARGS > '$log' 2>&1" || die "docker exec 启动失败"
        ;;
    *) die "未知 kind: $KIND" ;;
    esac

    info "等待健康检查 http://127.0.0.1:${PORT}${HEALTH}（最多 ${HEALTH_T}s）"
    if ! wait_health "$PORT" "$HEALTH" "$HEALTH_T"; then
        echo "  健康检查未通过，日志末尾：" >&2
        case "$KIND" in
        vllm) docker exec "$CONTAINER" bash -lc "tail -20 '${START_LOG:-/dev/null}'" >&2 || true ;;
        *) tail -20 /root/ascend-dev/logs/server.log >&2 || true ;;
        esac
        die "启动失败"
    fi
    info "健康检查通过"
    show_status
}

show_status() {
    echo "--- 运行状态 ------------------------------------------------"
    fnx_running  && echo "  fnx  : 运行中" || echo "  fnx  : 未运行"
    vllm_running && echo "  vllm : 运行中" || echo "  vllm : 未运行"
    # 探测配置里出现过的所有端口（通常只有 8000）
    local ports port mid h
    ports=$(python3 - "$CONF" <<'PY'
import json, sys
cfg = json.load(open(sys.argv[1], encoding="utf-8"))
ps = []
for m in (cfg.get("models") or {}).values():
    p = m.get("port", cfg.get("default_port", 8000))
    if p not in ps:
        ps.append(p)
print(" ".join(str(p) for p in ps))
PY
)
    for port in ${ports:-8000}; do
        mid=$(curl -s -m 4 "http://127.0.0.1:${port}/v1/models" 2>/dev/null | python3 -c \
            'import json,sys
try:
    d=json.load(sys.stdin); print(",".join(m["id"] for m in d.get("data",[])))
except Exception:
    pass' 2>/dev/null || true)
        if [ -n "$mid" ]; then
            echo "  /v1/models(:$port): $mid"
            h=$(curl -s -m 4 "http://127.0.0.1:${port}/health" 2>/dev/null || true)
            [ -n "$h" ] && echo "  /health(:$port): ${h:0:150}"
        fi
    done
    echo "  NPU 进程:"
    npu-smi info 2>/dev/null | grep -E "VLLMWorker|fnx|python" | sed 's/^/    /' | head -4
    echo "------------------------------------------------------------"
}

show_logs() {
    local name=$1 lines=${2:-40}
    eval "$(load_profile "$name")" || die "配置里没有模型 '$name'"
    case "$KIND" in
    vllm) docker exec "$CONTAINER" bash -lc "tail -n $lines '${START_LOG:-/dev/null}'" ;;
    *)    tail -n "$lines" /root/ascend-dev/logs/server.log ;;
    esac
}

# ---------- 入口 ----------
cmd=${1:-status}
case "$cmd" in
list)    echo "可用模型（配置 $CONF）："; list_models ;;
status)  show_status ;;
start)   [ $# -ge 2 ] || die "用法: modelctl.sh start <模型>"; start_model "$2" ;;
restart) [ $# -ge 2 ] || die "用法: modelctl.sh restart <模型>"; start_model "$2" ;;
stop)
    if [ $# -ge 2 ]; then
        eval "$(load_profile "$2")" || die "配置里没有模型 '$2'"
        case "$KIND" in
        fnx)  stop_fnx ;;
        vllm) stop_vllm ;;
        esac
        wait_npu_free "${STOP_T:-90}" || info "显存未完全释放"
        info "已停止 $2"
    else
        stop_all 90
        info "已停止全部模型"
    fi ;;
logs)    [ $# -ge 2 ] || die "用法: modelctl.sh logs <模型> [行数]"; show_logs "$2" "${3:-40}" ;;
*)       echo "用法: $0 {list|status|start <模型>|stop [模型]|restart <模型>|logs <模型> [行数]}"; exit 2 ;;
esac
