#!/bin/bash
# =============================================================================
# 30 · 部署服务（幂等）
#   - 写生产启动脚本 /root/ascend-dev/p275.sh（基线配置）
#   - 写看护脚本 /root/ascend-dev/watchdog.sh
#   - 安装 systemd 单元 fnx.service（开机自启 + 进程兜底）
#   - 启动并等待 /health
# =============================================================================
set -u
RUN=${RUN:-/root/ascend-dev}
CTN=${CTN:-fnx}
MDL=${MDL:-/models}
LOG=$RUN/logs/server.log
PY="$RUN/venv/bin/python"
[ -n "${LQP:-}" ] || LQP=$MDL/Qwen3.8-Flash-Next-LQP
mkdir -p "$RUN/logs"

echo "== 30 部署 =="

# ---------- 1. 生产启动脚本 ----------
cat > "$RUN/p275.sh" <<EOF
#!/bin/bash
# 生产启动脚本 —— 基线 baseline-r31（48 层：dev0 46 图捕获 + dev1 2 eager + head）
set -u
LOG=$LOG
STAMP=$RUN/logs/last_restart
WD=\${WD:-1}
pkill -f watchdog.sh 2>/dev/null || true
sleep 1
docker exec $CTN bash -lc "pkill -f '[p]ython -m fnx.server' || true"
for i in \$(seq 1 30); do
  docker exec $CTN bash -lc "pgrep -f '[p]ython -m fnx.server'" >/dev/null 2>&1 || break
  sleep 2
done
if docker exec $CTN bash -lc "pgrep -f '[p]ython -m fnx.server'" >/dev/null 2>&1; then
  docker exec $CTN bash -lc "pkill -9 -f '[p]ython -m fnx.server' || true"; sleep 5
fi
[ -f "\$LOG" ] && mv -f "\$LOG" "\$LOG.prev"
date +%s > "\$STAMP"
T0=\$(date +%s)
docker exec -d $CTN bash -lc "cd $RUN/fnx && \\
  FNX_PACK=0 FNX_EXP_G=64 FNX_MOE_KERNEL=0 FNX_NO_PLE=1 FNX_PACKED_TORCH=1 \\
  FNX_LPC=46 FNX_NLAYERS=48 FNX_PORT=8000 FNX_BATCH=1 \\
  FNX_PREFIX_CACHE=1 FNX_CONV_CACHE=4 FNX_PROF=0 FNX_ENABLE_THINKING=0 \\
  FNX_LQP=$LQP \\
  $PY -m fnx.server > \$LOG 2>&1"
H=""
for i in \$(seq 1 30); do
  sleep 10
  H=\$(curl -s -m 4 http://127.0.0.1:8000/health || true)
  [ -n "\$H" ] && { echo "ready 冷启动 = \$(( \$(date +%s) - T0 )) s  \$H"; break; }
  grep -qiE "out of memory|Tried to allocate" "\$LOG" 2>/dev/null && {
    echo "!! OOM"; grep -iE "Tried to allocate" "\$LOG" | tail -1; exit 1; }
done
[ -z "\$H" ] && { echo "!! 未就绪"; tail -8 "\$LOG"; exit 1; }
grep -E "装配 |图捕获 |停止符|前缀缓存|hbm[01]:" "\$LOG" | sed 's/^/  /'
if [ "\$WD" = 1 ]; then
  pkill -f watchdog.sh 2>/dev/null || true
  nohup bash $RUN/watchdog.sh > $RUN/logs/watchdog.log 2>&1 &
  sleep 1
  echo "  watchdog: \$(pgrep -af watchdog.sh | head -1)"
fi
EOF
chmod +x "$RUN/p275.sh"

# ---------- 2. 看护脚本 ----------
cat > "$RUN/watchdog.sh" <<EOF
#!/bin/bash
# 看护：20 s 探活；连续失败才重启；刚重启过有 150 s 宽限期；进程还活着时放宽到 15 次
set -u
LOG=$RUN/logs/watchdog.log
INTERVAL=\${WD_INTERVAL:-20}
FAILS=\${WD_FAILS:-3}
ALIVE_MAX=\${WD_ALIVE_MAX:-15}
GRACE=\${WD_GRACE:-150}
LOCK=/tmp/fnx_restart.lock
STAMP=$RUN/logs/last_restart
fail=0
echo "[\$(date '+%F %T')] watchdog 启动 pid=\$\$ interval=\${INTERVAL}s grace=\${GRACE}s" >> "\$LOG"
while true; do
  now=\$(date +%s); last=\$(cat "\$STAMP" 2>/dev/null || echo 0)
  if [ \$((now - last)) -lt "\$GRACE" ]; then fail=0; sleep "\$INTERVAL"; continue; fi
  H=\$(curl -s -m 6 http://127.0.0.1:8000/health 2>/dev/null || true)
  if echo "\$H" | grep -q '"status":"ok"'; then
    [ "\$fail" -gt 0 ] && echo "[\$(date '+%F %T')] 恢复（之前失败 \$fail 次）" >> "\$LOG"
    fail=0
  else
    fail=\$((fail+1))
    echo "[\$(date '+%F %T')] health 失败 \$fail: \${H:0:60}" >> "\$LOG"
    ALIVE=\$(docker exec $CTN bash -lc "pgrep -f '[p]ython -m fnx.server'" >/dev/null 2>&1 && echo 1 || echo 0)
    LIM=\$FAILS; [ "\$ALIVE" = 1 ] && LIM=\$ALIVE_MAX
    if [ "\$fail" -ge "\$LIM" ]; then
      if mkdir "\$LOCK" 2>/dev/null; then
        echo "[\$(date '+%F %T')] → 触发重启" >> "\$LOG"
        WD=0 bash $RUN/p275.sh >> "\$LOG" 2>&1
        rmdir "\$LOCK" 2>/dev/null || true
      fi
      fail=0
    fi
  fi
  sleep "\$INTERVAL"
done
EOF
chmod +x "$RUN/watchdog.sh"

# ---------- 3. systemd ----------
cat > /etc/systemd/system/fnx.service <<EOF
[Unit]
Description=fnx Qwen3.8-Flash-Next backend (2x910B4)
After=docker.service network-online.target
Requires=docker.service

[Service]
Type=simple
ExecStartPre=/bin/bash -c 'WD=0 /bin/bash $RUN/p275.sh'
ExecStart=/bin/bash $RUN/watchdog.sh
Restart=always
RestartSec=30
TimeoutStartSec=600

[Install]
WantedBy=multi-user.target
EOF
systemctl daemon-reload
systemctl enable fnx.service >/dev/null 2>&1 || true
echo "  systemd 单元已安装并 enable"

# ---------- 4. 启动 ----------
pkill -f watchdog.sh 2>/dev/null || true
systemctl restart fnx.service
H=""
for i in $(seq 1 30); do
  sleep 10
  H=$(curl -s -m 4 http://127.0.0.1:8000/health || true)
  [ -n "$H" ] && { echo "  /health → $H"; break; }
done
if [ -z "$H" ]; then echo "  ✗ 启动失败，看 $LOG"; tail -20 "$LOG"; exit 1; fi
systemctl is-enabled fnx.service | sed 's/^/  enabled: /'
systemctl is-active  fnx.service | sed 's/^/  active : /'
echo "  30 完成（服务已在 :8000 上，systemd 托管）"
