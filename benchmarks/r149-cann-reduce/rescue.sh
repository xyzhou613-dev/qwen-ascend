#!/bin/bash
set -u
cd /models/fnx-diagnostics/r149-cann-reduce-20261112
exec 9>rescue.lock
flock -n 9 || exit 0
test -f lease.active || exit 0
if [ -f controller.pid ];then
 task_pid=$(cat controller.pid)
 if [ "$task_pid" != "$PPID" ] && [ -r "/proc/$task_pid/cmdline" ] && tr '\0' ' ' < "/proc/$task_pid/cmdline" | grep -qF "$PWD/controller.sh";then kill -TERM "$task_pid" 2>/dev/null || true;fi
fi
docker exec bole-model-Flash /root/ascend-dev/venv/bin/python "$PWD/stop_owned.py"
canonical=0
if curl -fsS -m 5 http://127.0.0.1:9002/health > restored-health.json;then
 if python3 -c 'import json;h=json.load(open("restored-health.json"));assert h["service_revision"]=="r144" and h["worker_alive"] and not h.get("norm_pair",False)';then canonical=1;fi
fi
if [ "$canonical" != 1 ];then timeout 780 env WD=0 bash /root/ascend-dev/p275.sh >> restore.log 2>&1 || exit 1;fi
curl -fsS -m 5 http://127.0.0.1:9002/health > restored-health.json || exit 1
python3 -c 'import json;h=json.load(open("restored-health.json"));assert h["service_revision"]=="r144" and h["norm_post"] and h["delta_reuse"] and h["worker_alive"]' || exit 1
rm -f /run/systemd/system/model-switch.service.d/r149-exclusive.conf
systemctl daemon-reload
systemctl start model-switch.service
if ! pgrep -f '(^|/)bash /root/ascend-dev/watchdog.sh' >/dev/null;then systemd-run --unit=fnx-r149-watchdog --collect /bin/bash /root/ascend-dev/watchdog.sh;fi
rm -f lease.active original-private.json
echo CANONICAL_R144_MANAGER_RESTORED
