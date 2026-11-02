#!/bin/bash
set -euo pipefail
ROOT=/root/ascend-dev
grep -q R105_INVERSE_COMPLETE "$ROOT/logs/r105_inverse.log"
python3 "$ROOT/r106_prepare.py"
trap 'systemctl start fnx.service' ERR
systemctl stop fnx.service
docker exec fnx bash -lc "pkill -f '[p]ython -m fnx.server' || true"
sleep 3
docker exec fnx bash -lc 'set -a; source /root/ascend-dev/r106/candidate.env; set +a; export ASCEND_RT_VISIBLE_DEVICES=0; cd /root/ascend-dev/r106; /root/ascend-dev/venv/bin/python moe_test.py'
mkdir -p "$ROOT/r106/short" "$ROOT/r106/long" "$ROOT/r106/short_after"
docker exec -d fnx bash -lc 'set -a; source /root/ascend-dev/r106/candidate.env; set +a; cd /root/ascend-dev/r106; /root/ascend-dev/venv/bin/python -m fnx.server > /root/ascend-dev/logs/r106.server.log 2>&1'
ready=0
for i in $(seq 1 300);do
 sleep 2
 if curl -fs --max-time 3 http://127.0.0.1:8000/health > "$ROOT/r106/health.json";then ready=1;break;fi
 grep -qE 'Traceback|out of memory' "$ROOT/logs/r106.server.log" && break
done
[ "$ready" = 1 ]
docker exec fnx bash -lc 'cd /root/ascend-dev/r106; BENCH_DIR=/root/ascend-dev/r106/short /root/ascend-dev/venv/bin/python r51_payloads.py; BENCH_DIR=/root/ascend-dev/r106/short /root/ascend-dev/venv/bin/python r51_measure.py'
docker exec fnx bash -lc 'cd /root/ascend-dev/r106; TEST_CTX=262144 LONG_OUTPUT=128 LONG_EXPLAIN=1 LONG_CONTINUE=1 RESULT_DIR=/root/ascend-dev/r106/long /root/ascend-dev/venv/bin/python r52_long_client.py'
docker exec fnx bash -lc 'cd /root/ascend-dev/r106; BENCH_DIR=/root/ascend-dev/r106/short_after /root/ascend-dev/venv/bin/python r51_payloads.py; BENCH_DIR=/root/ascend-dev/r106/short_after /root/ascend-dev/venv/bin/python r51_measure.py'
curl -fs --max-time 5 http://127.0.0.1:8000/health > "$ROOT/r106/final-health.json"
echo R106_CANDIDATE_ACCEPT
