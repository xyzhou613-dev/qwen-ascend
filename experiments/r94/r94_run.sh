#!/bin/bash
set -euo pipefail
ROOT=/root/ascend-dev
grep -q R93_PROFILE_ACCEPT "$ROOT/logs/r93_profile.log"
trap 'systemctl start fnx.service' EXIT
systemctl stop fnx.service
docker exec fnx bash -lc "pkill -f '[p]ython -m fnx.server' || true"
sleep 3
docker exec fnx bash -lc 'set -a; source /root/ascend-dev/r92/candidate.env; set +a; export ASCEND_RT_VISIBLE_DEVICES=0; cd /root/ascend-dev/r94; /root/ascend-dev/venv/bin/python index_test.py'
