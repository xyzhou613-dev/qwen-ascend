#!/bin/bash
# ★ 把服务纳入 systemd（第30轮工程化）：开机自启 + 进程级兜底
#   之前的状态：server 是 `docker exec -d` 起的孤儿进程，**主机重启后不会自己回来**
#   （容器有 unless-stopped 会起来，但里面的 server 不会）。
#   设计：ExecStart = 先 p275.sh（WD=0，只起服务）→ 再把看护跑成主进程；
#   看护死了 → systemd Restart=always 重跑整个单元（等价于完整重启）。
set -u
UNIT=/etc/systemd/system/fnx.service
cat > "$UNIT" <<'EOF'
[Unit]
Description=fnx Qwen3.8-Flash-Next backend (2x910B4)
After=docker.service network-online.target
Requires=docker.service

[Service]
Type=simple
ExecStartPre=/bin/bash -c 'WD=0 /bin/bash /root/ascend-dev/p275.sh'
ExecStart=/bin/bash /root/ascend-dev/watchdog.sh
Restart=always
RestartSec=30
TimeoutStartSec=600

[Install]
WantedBy=multi-user.target
EOF

echo "== 单元文件 =="; cat "$UNIT" | sed 's/^/  /'
systemctl daemon-reload
systemctl enable fnx.service 2>&1 | sed 's/^/  enable: /'

# 停止手工看护，交给 systemd
pkill -f watchdog.sh 2>/dev/null || true
sleep 1
echo
echo "== 启动 =="
systemctl restart fnx.service
for i in $(seq 1 30); do
  sleep 10
  H=$(curl -s -m 4 http://127.0.0.1:8000/health || true)
  [ -n "$H" ] && { echo "  ready  $H"; break; }
done
[ -z "${H:-}" ] && { echo "  !! 未就绪"; systemctl status fnx.service --no-pager -l | tail -20; exit 1; }
echo
echo "== 状态 =="
systemctl is-enabled fnx.service 2>/dev/null | sed 's/^/  enabled: /'
systemctl is-active fnx.service 2>/dev/null | sed 's/^/  active: /'
systemctl status fnx.service --no-pager -l | head -12 | sed 's/^/  /'
echo
echo "== 看护进程（应只有 systemd 起的那一个）: $(pgrep -af 'watchdog.sh' | head -2 | tr '\n' ' ')"
