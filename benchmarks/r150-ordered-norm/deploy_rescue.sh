#!/bin/bash
set -u
cd /models/fnx-diagnostics/r150-ordered-norm-20261112
exec 9>deployment-rescue.lock
flock -n 9 || exit 0
test -f deployment.active || exit 0
if [ -f deployment.pid ];then
 task_pid=$(cat deployment.pid)
 if [ "$task_pid" != "$PPID" ] && [ -r "/proc/$task_pid/cmdline" ] && tr '\0' ' ' < "/proc/$task_pid/cmdline" | grep -qF "$PWD/deploy.sh";then kill -TERM "$task_pid" 2>/dev/null || true;fi
fi
docker exec bole-model-Flash /root/ascend-dev/venv/bin/python "$PWD/deploy_stop.py"
python3 deployment_files.py rollback || exit 1
timeout 780 env WD=0 bash /root/ascend-dev/p275.sh >> deployment-restore.log 2>&1 || exit 1
curl -fsS -m 5 http://127.0.0.1:9002/health > deployment-restored-health.json || exit 1
python3 -c 'import json;h=json.load(open("deployment-restored-health.json"));assert h["service_revision"]=="r144" and h["worker_alive"] and not h.get("norm_ordered",False)' || exit 1
rm -f /run/systemd/system/model-switch.service.d/r150-exclusive.conf
systemctl daemon-reload
systemctl start model-switch.service
if ! pgrep -f '(^|/)bash /root/ascend-dev/watchdog.sh' >/dev/null;then systemd-run --unit=fnx-r150-watchdog --collect /bin/bash /root/ascend-dev/watchdog.sh;fi
rm -f deployment.active
echo DEPLOYMENT_ROLLED_BACK_TO_R144
