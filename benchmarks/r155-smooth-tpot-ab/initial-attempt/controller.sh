#!/bin/bash
set -euo pipefail
cd /models/fnx-diagnostics/r155-smooth-tpot-ab-20261113
python3 idle_guard.py
docker exec bole-model-Flash /root/ascend-dev/venv/bin/python "$PWD/snapshot.py"
echo $$ > controller.pid
touch lease.active
cleanup() {
 set +e
 if bash "$PWD/rescue.sh" && test ! -f lease.active;then systemctl stop fnx-r155-rescue.timer;else echo RESTORE_PENDING_TIMER_RETAINED >&2;fi
}
trap cleanup EXIT
trap 'exit 143' TERM INT
systemd-run --unit=fnx-r155-rescue --on-active=45min --property=Restart=on-failure --property=RestartSec=30 /bin/bash "$PWD/rescue.sh"
systemctl stop model-switch.service
mkdir -p /run/systemd/system/model-switch.service.d
printf '[Unit]\nRefuseManualStart=yes\n' > /run/systemd/system/model-switch.service.d/r155-exclusive.conf
systemctl daemon-reload
if pgrep -f '(^|/)bash /root/ascend-dev/watchdog.sh' > watchdog.pids;then while read -r pid;do kill -TERM "$pid";done < watchdog.pids;fi
python3 idle_guard.py
touch stop-requested
docker exec bole-model-Flash /root/ascend-dev/venv/bin/python "$PWD/stop_owned.py"
for phase in baseline candidate rollback;do
 docker exec bole-model-Flash /root/ascend-dev/venv/bin/python "$PWD/stop_owned.py"
 date +%s > "${phase}-start-epoch.txt"
 docker exec bole-model-Flash /root/ascend-dev/venv/bin/python "$PWD/launch.py" "$phase"
 ready=0
 for attempt in $(seq 1 420);do
  sleep 2
  if curl -fs -m 2 http://127.0.0.1:19002/health > "${phase}-health.json";then ready=1;break;fi
  if grep -qE 'Traceback \(most recent call last\)|out of memory|ERR99999' "${phase}-server.log";then echo STARTUP_FAILED;tail -n 20 "${phase}-server.log";exit 3;fi
 done
 test "$ready" = 1
 python3 check_ready.py "$phase"
 python3 http_bench.py "$phase"
 docker exec bole-model-Flash /root/ascend-dev/venv/bin/python "$PWD/stop_owned.py"
done
bash "$PWD/rescue.sh"
test ! -f lease.active
systemctl stop fnx-r155-rescue.timer
python3 formal_smoke.py
python3 evaluate.py
echo TPOT_AB_RESTORE_COMPLETE
