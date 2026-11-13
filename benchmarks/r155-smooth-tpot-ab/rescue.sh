#!/bin/bash
set -u
cd /models/fnx-diagnostics/r155-smooth-tpot-ab-20261113
exec 9>rescue.lock
flock -n 9 || exit 0
test -f lease.active || exit 0
if [ -f controller.pid ];then
 pid=$(cat controller.pid)
 if [ "$pid" != "$PPID" ] && [ -r "/proc/$pid/cmdline" ] && tr '\0' ' ' < "/proc/$pid/cmdline" | grep -qF "$PWD/controller.sh";then kill -TERM "$pid" 2>/dev/null || true;fi
fi
docker exec bole-model-Flash /root/ascend-dev/venv/bin/python "$PWD/stop_owned.py" || exit 1
canonical=0
if curl -fsS -m 5 http://127.0.0.1:9002/health > restored-health.json;then
 if python3 -c 'import json;h=json.load(open("restored-health.json"));assert h["service_revision"]=="r152" and h["worker_alive"] and not h.get("dense_w8a8_enabled",False)';then canonical=1;fi
fi
if [ "$canonical" != 1 ];then timeout 800 env WD=0 bash /root/ascend-dev/p275.sh >> restore.log 2>&1 || exit 1;fi
curl -fsS -m 5 http://127.0.0.1:9002/health > restored-health.json || exit 1
python3 -c 'import json;h=json.load(open("restored-health.json"));assert h["service_revision"]=="r152" and h["worker_alive"] and h["ctx"]==262144 and h["state_update"] and h["norm_ordered"]' || exit 1
rm -f /run/systemd/system/model-switch.service.d/r155-exclusive.conf
systemctl daemon-reload
systemctl start model-switch.service || exit 1
if ! pgrep -f '(^|/)bash /root/ascend-dev/watchdog.sh' >/dev/null;then systemd-run --unit=fnx-r155-watchdog --collect /bin/bash /root/ascend-dev/watchdog.sh || exit 1;fi
rm -f lease.active original-private.json
echo CANONICAL_R152_RESTORED
