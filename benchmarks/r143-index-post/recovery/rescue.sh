#!/bin/bash
set -u
cd /models/fnx-diagnostics/r143-index-post-20261112
exec 9>rescue.lock
flock -n 9 || exit 0
test -f lease.active || exit 0
if [ -f controller.pid ];then
 task_controller=$(cat controller.pid)
 if [ "$task_controller" != "$PPID" ] && [ -r "/proc/$task_controller/cmdline" ];then
  if tr '\0' ' ' < "/proc/$task_controller/cmdline" | grep -qF '/models/fnx-diagnostics/r143-index-post-20261112/';then kill -TERM "$task_controller" 2>/dev/null || true;fi
 fi
fi
docker exec bole-model-Flash /root/ascend-dev/venv/bin/python "$PWD/rescue_stop.py"
if [ -f deployment-backup/manifest.json ];then python3 rollback_files.py || exit 1;fi
docker exec bole-model-Flash /root/ascend-dev/venv/bin/python "$PWD/restore_original.py"
ready=0
for i in $(seq 1 300);do
 if curl -fs --max-time 2 http://127.0.0.1:9002/health > rescue-health.json;then
  if python3 -c 'import json; h=json.load(open("rescue-health.json")); assert h["service_revision"]=="r142" and h["worker_alive"]' 2>/dev/null;then ready=1;break;fi
 fi
 sleep 2
done
if [ "$ready" != 1 ];then
 timeout 720 env WD=1 bash /root/ascend-dev/p275.sh >> rescue-canonical.log 2>&1 || true
 curl -fs --max-time 3 http://127.0.0.1:9002/health > rescue-health.json || true
fi
rm -f /run/systemd/system/model-switch.service.d/r143-exclusive.conf /run/systemd/system/model-switch.service.d/r143-deployment.conf
systemctl daemon-reload
systemctl start model-switch.service
if [ -f watchdog.was-running ] && ! pgrep -f '^bash /root/ascend-dev/watchdog.sh$' >/dev/null;then
 nohup bash /root/ascend-dev/watchdog.sh > /root/ascend-dev/logs/watchdog.log 2>&1 < /dev/null &
fi
curl -fs --max-time 3 http://127.0.0.1:9002/health > rescue-health.json || exit 1
python3 -c 'import json;h=json.load(open("rescue-health.json"));assert h["service_revision"]=="r142" and h["worker_alive"]' || exit 1
rm -f lease.active
echo R142_AND_MANAGER_RESTORED
