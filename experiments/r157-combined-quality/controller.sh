#!/bin/bash
set -euo pipefail
P=/models/fnx-diagnostics/r157-combined-quality-20261113
cd "$P"
python3 idle_guard.py
docker exec bole-model-Flash /root/ascend-dev/venv/bin/python "$P/snapshot.py"
cat /proc/sys/kernel/random/boot_id > original-boot-id
echo $$ > controller.pid
if [ ! -f window-start.txt ];then date +%s > window-start.txt;fi
python3 -c 'import uuid;print(uuid.uuid4().hex)' > run-id
export R157_RUN_ID=$(cat run-id)
touch lease.active
cleanup(){ set +e;if bash "$P/rescue.sh" && test ! -f lease.active;then if systemctl is-active --quiet fnx-r157-rescue.timer;then systemctl stop fnx-r157-rescue.timer;fi;else echo RESTORE_PENDING_GUARDIAN_RETAINED >&2;fi; }
trap cleanup EXIT
trap 'exit 143' TERM INT
# Leave at least 25 minutes of the approved six hours for cleanup/recovery.
remaining=$((20100-$(date +%s)+$(cat window-start.txt)))
test "$remaining" -gt 1800
systemd-run --unit=fnx-r157-rescue --on-active="${remaining}s" --property=Restart=on-failure --property=RestartSec=20 /bin/bash "$P/rescue.sh"
systemctl stop model-switch.service
mkdir -p /run/systemd/system/model-switch.service.d
printf '[Unit]\nRefuseManualStart=yes\n' > /run/systemd/system/model-switch.service.d/r157-exclusive.conf
systemctl daemon-reload
if pgrep -f '(^|/)bash /root/ascend-dev/watchdog.sh' > watchdog.pids;then while read -r pid;do kill -TERM "$pid";done < watchdog.pids;fi
python3 idle_guard.py
touch stop-requested
docker exec bole-model-Flash /root/ascend-dev/venv/bin/python "$P/stop_owned.py"
test -s baseline/summary.json
for phase in smooth combined rollback;do
 docker exec bole-model-Flash /root/ascend-dev/venv/bin/python "$P/stop_owned.py"
 date +%s > "$phase-start.txt"
 docker exec bole-model-Flash /root/ascend-dev/venv/bin/python "$P/launch.py" "$phase"
 ready=0
 for attempt in $(seq 1 420);do
  sleep 2
  if curl -fs -m 2 http://127.0.0.1:19002/health > "$phase-health.json";then ready=1;break;fi
  if grep -qE 'Traceback \(most recent call last\)|out of memory|ERR99999' "$phase-server.log";then echo STARTUP_FAILED;tail -n 18 "$phase-server.log";exit 3;fi
 done
 test "$ready" = 1
 python3 check_ready.py "$phase"
 python3 http_bench.py "$phase"
 if [ "$phase" = combined ] || [ "$phase" = rollback ];then
  python3 quality_client.py "$phase" --full
  python3 quality_eval.py --phase "$phase" --url http://127.0.0.1:19002 --expected-variant "$phase"
 else
  python3 quality_client.py "$phase"
 fi
 docker exec bole-model-Flash /root/ascend-dev/venv/bin/python "$P/stop_owned.py"
done
# Profiling reuses a single exclusive Engine only when sufficient budget remains.
if [ $(( $(date +%s)-$(cat window-start.txt) )) -lt 18000 ];then if ! bash "$P/profile.sh";then echo OPTIONAL_PROFILE_FAILED_QUALITY_PRESERVED;fi;fi
bash "$P/rescue.sh"
test ! -f lease.active
if systemctl is-active --quiet fnx-r157-rescue.timer;then systemctl stop fnx-r157-rescue.timer;fi
python3 formal_smoke.py
echo R157_FULL_RUN_AND_RESTORE_COMPLETE
