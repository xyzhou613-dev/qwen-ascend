#!/bin/bash
set -euo pipefail
ROOT=/root/ascend-dev
test -f "$ROOT/r56/pre_optimization.tgz"
systemctl stop fnx.service
docker exec fnx bash -lc "pkill -f '[p]ython -m fnx.server' || true"
tar -xzf "$ROOT/r56/pre_optimization.tgz" -C "$ROOT"
systemctl start fnx.service
curl -fs --max-time 5 http://127.0.0.1:8000/health
