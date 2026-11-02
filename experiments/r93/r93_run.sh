#!/bin/bash
set -euo pipefail
ROOT=/root/ascend-dev
python3 "$ROOT/r93_prepare.py"
cp "$ROOT/r93_profile.py" "$ROOT/r93/profile_current.py"
trap 'systemctl start fnx.service' EXIT
systemctl stop fnx.service
docker exec fnx bash -lc "pkill -f '[p]ython -m fnx.server' || true"
sleep 3
docker exec fnx bash -lc 'set -a; source /root/ascend-dev/r93/profile.env; set +a; cd /root/ascend-dev/r93; /root/ascend-dev/venv/bin/python profile_current.py'
