#!/bin/bash
set -u
P=/models/fnx-diagnostics/r157-combined-quality-20261113
cd "$P"
exec 9>rescue.lock
flock -n 9 || exit 0
test -f lease.active || exit 0
if [ -f controller.pid ];then
 pid=$(cat controller.pid)
 if [ "$pid" != "$PPID" ] && [ -r "/proc/$pid/cmdline" ] && tr '\0' ' ' < "/proc/$pid/cmdline" | grep -qF "$P/controller.sh";then kill -TERM "$pid" 2>/dev/null || true;fi
fi
# Only judges created by this batch, never arbitrary containers.
if [ -f run-id ];then
 for cid in $(docker ps -q --filter "label=fnx.diagnostic=r157" --filter "label=fnx.run=$(cat run-id)");do docker rm -f "$cid" >/dev/null 2>&1 || true;done
fi
docker exec bole-model-Flash /root/ascend-dev/venv/bin/python "$P/stop_owned.py" || exit 1
canonical=0
if curl -fsS -m 5 http://127.0.0.1:9002/health > restored-health.json;then
 if python3 -c 'import json;h=json.load(open("restored-health.json"));assert h["service_revision"]=="r152" and h["worker_alive"] and not h.get("diagnostic_variant")';then canonical=1;fi
fi
if [ "$canonical" != 1 ];then timeout 800 env WD=0 bash /root/ascend-dev/p275.sh >> restore.log 2>&1 || exit 1;fi
curl -fsS -m 5 http://127.0.0.1:9002/health > restored-health.json || exit 1
python3 -c 'import json;h=json.load(open("restored-health.json"));assert h["service_revision"]=="r152" and h["worker_alive"] and h["ctx"]==262144' || exit 1
rm -f /run/systemd/system/model-switch.service.d/r157-exclusive.conf
systemctl daemon-reload
systemctl start model-switch.service || exit 1
if ! pgrep -f '(^|/)bash /root/ascend-dev/watchdog.sh' >/dev/null;then systemd-run --unit=fnx-watchdog --collect /bin/bash /root/ascend-dev/watchdog.sh || exit 1;fi
rm -f lease.active original-private.json
echo CANONICAL_R152_RESTORED
