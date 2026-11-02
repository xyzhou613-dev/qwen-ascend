#!/bin/bash
set -euo pipefail
ROOT=/root/ascend-dev
ready=0
for i in $(seq 1 360);do
 if grep -q R109_PRODUCTION_ACCEPT "$ROOT/logs/r109_production.log";then ready=1;break;fi
 sleep 5
done
[ "$ready" = 1 ]
docker exec fnx bash -lc 'cd /root/ascend-dev/r109; /root/ascend-dev/venv/bin/python /root/ascend-dev/r109_sustained.py' > "$ROOT/logs/r109_sustained.log" 2>&1
curl -fs --max-time 5 http://127.0.0.1:8000/health > "$ROOT/r109/post-sustained-health.json"
echo R109_SUSTAINED_RUN_ACCEPT
