#!/bin/bash
set -euo pipefail
ROOT=/root/ascend-dev
test -f "$ROOT/r92/predeploy.tgz"
systemctl stop fnx.service
docker exec fnx bash -lc "pkill -f '[p]ython -m fnx.server' || true"
tar -xzf "$ROOT/r92/predeploy.tgz" -C "$ROOT"
# This file is introduced by r92; the restored pipeline does not import it.
rm -f "$ROOT/fnx/fnx/peer_ipc.py"
systemctl start fnx.service
curl -fsS -m 5 http://127.0.0.1:8000/health
