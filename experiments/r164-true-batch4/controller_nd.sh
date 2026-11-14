#!/bin/bash
set -euo pipefail
P=/models/fnx-diagnostics/r164-true-batch4-20261114
cd "$P"
cleanup(){ systemctl start --no-block fnx-r164-rescue.service; }
trap cleanup EXIT
trap 'exit 143' TERM INT
python3 idle_guard.py
docker exec bole-model-Flash /root/ascend-dev/venv/bin/python "$P/snapshot.py"
cat /proc/sys/kernel/random/boot_id > original-boot-id
systemctl stop model-switch.service
mkdir -p /run/systemd/system/model-switch.service.d
printf '[Unit]\nRefuseManualStart=yes\n' > /run/systemd/system/model-switch.service.d/r164-exclusive.conf
systemctl daemon-reload
python3 pause_watchdog.py
python3 idle_guard.py
touch stop-requested
docker exec bole-model-Flash /root/ascend-dev/venv/bin/python "$P/stop_owned.py"
for batch in 4 1;do
 test $(( $(date +%s)-$(cat window-start.txt) )) -lt 2100
 docker exec bole-model-Flash /root/ascend-dev/venv/bin/python "$P/launch_nd.py" "$batch"
done
echo ND_TRUE_BATCH4_AND_MATCHED_BATCH1_COMPLETED
