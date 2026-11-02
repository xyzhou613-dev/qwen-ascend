#!/bin/bash
set -euo pipefail
ROOT=/root/ascend-dev
ready=0
for i in $(seq 1 360);do
 if grep -q R109_SUSTAINED_RUN_ACCEPT "$ROOT/logs/r109_sustained_run.log";then ready=1;break;fi
 sleep 5
done
[ "$ready" = 1 ]
python3 - <<'PY'
from pathlib import Path
r=Path('/root/ascend-dev')
s=(r/'r108/r108_probe.py').read_text().replace('/root/ascend-dev/r108/libfnx_delta.so','/root/ascend-dev/r109/build/libfnx_delta.so')
s=s.replace('/root/ascend-dev/r108/probe.json','/root/ascend-dev/r109/rebuilt-probe.json').replace('R108_PROBE_COMPLETE','R109_REBUILT_ACCEPT')
(r/'r109/rebuilt_probe.py').write_text(s)
PY
docker exec fnx bash -lc 'cd /root/ascend-dev/r109; ASCEND_RT_VISIBLE_DEVICES=0 /root/ascend-dev/venv/bin/python rebuilt_probe.py' > "$ROOT/logs/r109_rebuilt_probe.log" 2>&1
grep -q R109_REBUILT_ACCEPT "$ROOT/logs/r109_rebuilt_probe.log"
curl -fs --max-time 5 http://127.0.0.1:8000/health > "$ROOT/r109/final-health.json"
echo R109_REBUILD_ACCEPT
