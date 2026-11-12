#!/bin/bash
set -u
cd /models/fnx-diagnostics/r144-raw-norm-20261112
exec 9>deployment-rescue.lock
flock -n 9 || exit 0
test -f deployment.active || exit 0
if [ -f deployment.pid ];then
 task_pid=$(cat deployment.pid)
 if [ "$task_pid" != "$PPID" ] && [ -r "/proc/$task_pid/cmdline" ] && tr '\0' ' ' < "/proc/$task_pid/cmdline" | grep -qF "$PWD/deployment.sh";then kill -TERM "$task_pid" 2>/dev/null || true;fi
fi
docker exec bole-model-Flash /root/ascend-dev/venv/bin/python "$PWD/rescue_stop.py"
if [ -f deployment-backup/manifest.json ];then python3 deployment_files.py rollback || exit 1;fi
timeout 780 env WD=1 bash /root/ascend-dev/p275.sh >> deployment-restore.log 2>&1
rc=$?
rm -f /run/systemd/system/model-switch.service.d/r144-exclusive.conf
systemctl daemon-reload
systemctl start model-switch.service
test "$rc" = 0 || exit 1
curl -fs -m 5 http://127.0.0.1:9002/health > deployment-restore-health.json || exit 1
bash "$PWD/restore_watchdog.sh"
rm -f deployment.active
echo DEPLOYMENT_RESCUED
