#!/bin/bash
set -u
cd /models/fnx-diagnostics/r145-hardware-profile-20261112
exec 9>rescue.lock
flock -n 9 || exit 0
test -f lease.active || exit 0
if [ -f controller.pid ];then
 task_pid=$(cat controller.pid)
 if [ "$task_pid" != "$PPID" ] && [ -r "/proc/$task_pid/cmdline" ] && tr '\0' ' ' < "/proc/$task_pid/cmdline" | grep -qF "$PWD/controller.sh";then kill -TERM "$task_pid" 2>/dev/null || true;fi
fi
docker exec bole-model-Flash /root/ascend-dev/venv/bin/python "$PWD/stop_diagnostic.py"
timeout 780 env WD=0 bash /root/ascend-dev/p275.sh >> restore.log 2>&1
rc=$?
rm -f /run/systemd/system/model-switch.service.d/r145-profile.conf
systemctl daemon-reload
systemctl start model-switch.service
test "$rc" = 0 || exit 1
curl -fs --max-time 5 http://127.0.0.1:9002/health > restored-health.json || exit 1
python3 -c 'import json;h=json.load(open("restored-health.json"));assert h["service_revision"]=="r144" and h["norm_post"] and h["delta_reuse"] and h["worker_alive"]' || exit 1
# A controller's background children die with its cgroup. This watchdog owns
# an independent transient service until the final SSH-side handback.
systemd-run --unit=fnx-profile-watchdog --collect /bin/bash /root/ascend-dev/watchdog.sh
rm -f lease.active
echo CANONICAL_R144_MANAGER_WATCHDOG_RESTORED
