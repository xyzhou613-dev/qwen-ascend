#!/bin/bash
set -u
cd /models/fnx-diagnostics/r144-raw-norm-20261112
exec 9>rescue.lock
flock -n 9 || exit 0
test -f lease.active || exit 0
if [ -f controller.pid ]; then
 task_pid=$(cat controller.pid)
 if [ "$task_pid" != "$PPID" ] && [ -r "/proc/$task_pid/cmdline" ] && tr '\0' ' ' < "/proc/$task_pid/cmdline" | grep -qF "$PWD/controller.sh"; then kill -TERM "$task_pid" 2>/dev/null || true;fi
fi
docker exec bole-model-Flash /root/ascend-dev/venv/bin/python "$PWD/rescue_stop.py"
if grep -q CANDIDATE_TESTS_COMPLETE controller.log;then
 timeout 180 docker exec -e ACCURACY_ONLY=1 bole-model-Flash /root/ascend-dev/venv/bin/python "$PWD/launch.py" full_layer_probe.py > full-layer-bits.log 2>&1 || true
fi
docker exec bole-model-Flash /root/ascend-dev/venv/bin/python "$PWD/rescue_stop.py"
if grep -q CANDIDATE_TESTS_COMPLETE controller.log && [ -f deploy-on-accept ];then
 if grep -q FULL_LAYER_COMPLETE full-layer-bits.log;then
  env R144_HANDOFF=1 bash "$PWD/deployment.sh" > deployment.log 2>&1
  rc=$?
  if [ "$rc" = 0 ];then
   rm -f lease.active
   echo R144_PRODUCTION_HANDOFF_COMPLETE
   exit 0
  fi
 fi
fi
timeout 780 env WD=1 bash /root/ascend-dev/p275.sh >> "$PWD/restore.log" 2>&1
ready=$?
rm -f /run/systemd/system/model-switch.service.d/r144-exclusive.conf
systemctl daemon-reload
systemctl start model-switch.service
if [ "$ready" != 0 ];then echo CANONICAL_RESTORE_FAILED;exit 1;fi
curl -fs --max-time 5 http://127.0.0.1:9002/health > restored-health.json || exit 1
python3 -c 'import json;h=json.load(open("restored-health.json"));assert h["service_revision"]=="r143" and h["worker_alive"]' || exit 1
bash "$PWD/restore_watchdog.sh"
rm -f lease.active
echo R143_CANONICAL_AND_MANAGER_RESTORED
