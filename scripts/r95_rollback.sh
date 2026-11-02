#!/bin/bash
set -euo pipefail
ROOT=/root/ascend-dev
test -f "$ROOT/r95/predeploy.tgz"
systemctl stop fnx.service
docker exec fnx bash -lc "pkill -f '[p]ython -m fnx.server' || true"
tar -xzf "$ROOT/r95/predeploy.tgz" -C "$ROOT"
rm -f "$ROOT/fnx/fnx/snapshot.py"
systemctl start fnx.service
curl -fsS --max-time 5 http://127.0.0.1:8000/health
