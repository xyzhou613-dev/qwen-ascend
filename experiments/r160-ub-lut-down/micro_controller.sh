#!/bin/bash
set -euo pipefail
P=/models/fnx-diagnostics/r160-ub-lut-down-20261114
cd "$P"
cleanup(){ systemctl start --no-block fnx-r160-rescue.service; }
trap cleanup EXIT
trap 'exit 143' TERM INT
python3 idle_guard.py
docker exec bole-model-Flash /root/ascend-dev/venv/bin/python "$P/snapshot.py"
cat /proc/sys/kernel/random/boot_id > original-boot-id
systemctl stop model-switch.service
mkdir -p /run/systemd/system/model-switch.service.d
printf '[Unit]\nRefuseManualStart=yes\n' > /run/systemd/system/model-switch.service.d/r160-exclusive.conf
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
python3 - <<'PY'
import re,subprocess
assert not re.findall(r'\|\s+\d+\s+\d+\s+\|\s+(\d+)\s+\|',subprocess.check_output(['npu-smi','info'],text=True))
PY
for d in 0 1;do
 timeout 170 docker exec bole-model-Flash /root/ascend-dev/venv/bin/python "$P/launch.py" "$d" --timing > "timing-device$d.log" 2>&1
done
echo TWO_CARD_LUT_COMPONENT_TIMING_COMPLETE_NO_MODEL_TEST
