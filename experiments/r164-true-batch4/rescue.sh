#!/bin/bash
set -euo pipefail
P=/models/fnx-diagnostics/r164-true-batch4-20261114
cd "$P"
exec 9>rescue.lock
flock -n 9 || exit 0
test -f lease.active || exit 0
if systemctl is-active --quiet fnx-r164-run.service;then systemctl stop fnx-r164-run.service || true;fi
docker exec bole-model-Flash /root/ascend-dev/venv/bin/python "$P/stop_owned.py"
canonical=0
if curl -fsS -m 5 http://127.0.0.1:9002/health > restored-health.json;then
 if python3 -c 'import json;h=json.load(open("restored-health.json"));assert h["service_revision"]=="r152" and h["worker_alive"] and not h.get("diagnostic_variant")';then canonical=1;fi
fi
if [ "$canonical" != 1 ];then timeout 800 env WD=0 bash /root/ascend-dev/p275.sh >> restore.log 2>&1;fi
curl -fsS -m 5 http://127.0.0.1:9002/health > restored-health.json
python3 -c 'import json;h=json.load(open("restored-health.json"));assert h["status"]=="ok" and h["worker_alive"] and h["service_revision"]=="r152" and h["ctx"]==262144 and "diagnostic_variant" not in h'
rm -f /run/systemd/system/model-switch.service.d/r164-exclusive.conf
systemctl daemon-reload
systemctl start model-switch.service
if ! pgrep -f '(^|/)bash /root/ascend-dev/watchdog.sh' >/dev/null;then
 systemd-run --unit=fnx-watchdog --collect /bin/bash /root/ascend-dev/watchdog.sh
fi
python3 - <<'PY'
import json,time
from pathlib import Path
p=Path('.');start=int((p/'window-start.txt').read_text());end=time.time()
(p/'restore-complete.json').write_text(json.dumps(dict(epoch=end,elapsed_s=end-start,within_hour=end-start<=3600),indent=2))
PY
rm -f lease.active original-private.json
echo CANONICAL_R152_RESTORED
