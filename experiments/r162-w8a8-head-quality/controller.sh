#!/bin/bash
set -euo pipefail
P=/models/fnx-diagnostics/r162-w8a8-head-quality-20261114
cd "$P"
cleanup(){ systemctl start --no-block fnx-r162-rescue.service; }
trap cleanup EXIT
trap 'exit 143' TERM INT
python3 idle_guard.py
docker exec bole-model-Flash /root/ascend-dev/venv/bin/python "$P/snapshot.py"
cat /proc/sys/kernel/random/boot_id > original-boot-id
export R157_RUN_ID=$(cat run-id)
export BENCH_URL=http://127.0.0.1:19002
systemctl stop model-switch.service
mkdir -p /run/systemd/system/model-switch.service.d
printf '[Unit]\nRefuseManualStart=yes\n' > /run/systemd/system/model-switch.service.d/r162-exclusive.conf
systemctl daemon-reload
python3 - <<'PY'
import os,signal
from pathlib import Path
for p in Path('/proc').glob('[0-9]*'):
 try:
  argv=p.joinpath('cmdline').read_bytes().split(b'\0')
  if argv and Path(os.fsdecode(argv[0])).name in ('bash','sh') and b'/root/ascend-dev/watchdog.sh' in argv:
   os.kill(int(p.name),signal.SIGTERM)
 except (FileNotFoundError,PermissionError,ProcessLookupError):pass
PY
python3 idle_guard.py
touch stop-requested
docker exec bole-model-Flash /root/ascend-dev/venv/bin/python "$P/stop_owned.py"
for phase in rollback smooth combined;do
 test $(( $(date +%s)-$(cat window-start.txt) )) -lt 18900
 date +%s > "$phase-start.txt"
 docker exec bole-model-Flash /root/ascend-dev/venv/bin/python "$P/launch.py" "$phase"
 ready=0
 for attempt in $(seq 1 420);do
  sleep 2
  if curl -fsS -m 2 http://127.0.0.1:19002/health > "$phase-health.json";then ready=1;break;fi
  if grep -qE 'Traceback \(most recent call last\)|out of memory|ERR99999' "$phase-server.log";then echo STARTUP_FAILED;exit 3;fi
 done
 test "$ready" = 1
 python3 check_ready.py "$phase"
 python3 http_bench.py "$phase"
 if [ "$phase" = rollback ];then
  python3 reuse_check.py
  python3 quality_client.py rollback --full
  python3 verify_old_prefill.py
  python3 quality_eval.py --phase rollback --url http://127.0.0.1:19002 --expected-variant rollback
 elif [ "$phase" = smooth ];then
  python3 quality_client.py smooth
 else
  python3 quality_client.py combined
  python3 short_gate.py
  python3 quality_client.py combined --full
  python3 quality_eval.py --phase combined --url http://127.0.0.1:19002 --expected-variant combined
 fi
 docker exec bole-model-Flash /root/ascend-dev/venv/bin/python "$P/stop_owned.py"
done
echo COMPLETE_HEAD_CANDIDATE_PAIRING_PENDING_RESTORE_AND_ANALYSIS
