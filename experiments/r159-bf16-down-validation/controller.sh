#!/bin/bash
set -euo pipefail
P=/models/fnx-diagnostics/r159-bf16-down-validation-20261114
cd "$P"
cleanup(){ systemctl start --no-block fnx-r159-rescue.service; }
trap cleanup EXIT
trap 'exit 143' TERM INT
python3 idle_guard.py
docker exec bole-model-Flash /root/ascend-dev/venv/bin/python "$P/snapshot.py"
cat /proc/sys/kernel/random/boot_id > original-boot-id
systemctl stop model-switch.service
mkdir -p /run/systemd/system/model-switch.service.d
printf '[Unit]\nRefuseManualStart=yes\n' > /run/systemd/system/model-switch.service.d/r159-exclusive.conf
systemctl daemon-reload
python3 - <<'PY'
import os,signal
from pathlib import Path
for p in Path('/proc').glob('[0-9]*'):
 try:
  cmd=p.joinpath('cmdline').read_bytes().split(b'\0')
  if cmd and Path(os.fsdecode(cmd[0])).name in ('bash','sh') and b'/root/ascend-dev/watchdog.sh' in cmd:
   os.kill(int(p.name),signal.SIGTERM)
 except (FileNotFoundError,PermissionError,ProcessLookupError):pass
PY
python3 idle_guard.py
touch stop-requested
docker exec bole-model-Flash /root/ascend-dev/venv/bin/python "$P/stop_owned.py"
for d in 0 1;do
 timeout 420 docker exec bole-model-Flash /root/ascend-dev/venv/bin/python "$P/launch_component.py" "$d" component_probe.py > "component-device$d.log" 2>&1
done
python3 select_component.py
for d in 0 1;do
 timeout 570 docker exec bole-model-Flash /root/ascend-dev/venv/bin/python "$P/launch_component.py" "$d" layer_probe.py > "layers-device$d.log" 2>&1
done
python3 gate_layers.py
for phase in baseline combined;do
 test $(( $(date +%s)-$(cat window-start.txt) )) -lt 1500
 date +%s > "$phase-start.txt"
 docker exec bole-model-Flash /root/ascend-dev/venv/bin/python "$P/launch.py" "$phase"
 ready=0
 for attempt in $(seq 1 360);do
  sleep 2
  if curl -fsS -m 2 http://127.0.0.1:19002/health > "$phase-health.json";then ready=1;break;fi
  if grep -qE 'Traceback \(most recent call last\)|out of memory|ERR99999' "$phase-server.log";then echo STARTUP_FAILED;exit 3;fi
 done
 test "$ready" = 1
 python3 check_ready.py "$phase"
 python3 verify_rounded_audit.py "$phase"
 python3 quality_screen.py "$phase"
 python3 http_bench.py "$phase"
 docker exec bole-model-Flash /root/ascend-dev/venv/bin/python "$P/stop_owned.py"
done
echo BOUNDED_SCREEN_AND_HTTP_COMPLETE_NOT_FULL_ACCEPTANCE
