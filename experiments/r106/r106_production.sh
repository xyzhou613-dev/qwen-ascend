#!/bin/bash
set -euo pipefail
ROOT=/root/ascend-dev
python3 "$ROOT/r106_deploy.py"
trap 'bash /root/ascend-dev/r106_rollback.sh' ERR
mkdir -p "$ROOT/r106/production256" "$ROOT/r106/production8k" "$ROOT/r106/production256_second"
curl -fs --max-time 5 http://127.0.0.1:8000/health > "$ROOT/r106/production-health-start.json"
docker exec fnx bash -lc 'cd /root/ascend-dev/r106; TEST_CTX=262144 LONG_OUTPUT=128 LONG_EXPLAIN=1 LONG_CONTINUE=1 RESULT_DIR=/root/ascend-dev/r106/production256 /root/ascend-dev/venv/bin/python r52_long_client.py'
docker exec fnx bash -lc 'cd /root/ascend-dev/r106; BENCH_DIR=/root/ascend-dev/r106/production8k /root/ascend-dev/venv/bin/python r51_payloads.py; BENCH_DIR=/root/ascend-dev/r106/production8k /root/ascend-dev/venv/bin/python r51_measure.py'
docker exec fnx bash -lc 'cd /root/ascend-dev/r106; TEST_CTX=262144 LONG_OUTPUT=128 LONG_EXPLAIN=1 LONG_CONTINUE=1 RESULT_DIR=/root/ascend-dev/r106/production256_second /root/ascend-dev/venv/bin/python r52_long_client.py'
curl -fs --max-time 5 http://127.0.0.1:8000/health > "$ROOT/r106/production-health-final.json"
curl -fs --max-time 5 http://127.0.0.1:8000/fnx/prof > "$ROOT/r106/production-prof-final.json"
systemctl is-active fnx.service
systemctl is-enabled fnx.service
echo R106_PRODUCTION_ACCEPT
