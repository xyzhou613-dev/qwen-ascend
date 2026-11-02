#!/bin/bash
set -euo pipefail
ROOT=/root/ascend-dev
grep -q R108_PROBE_COMPLETE "$ROOT/logs/r108_probe_final.log"
python3 "$ROOT/r109_prepare.py"
cp "$ROOT/r109_delta_test.py" "$ROOT/r109/delta_test.py"
trap 'systemctl start fnx.service' ERR
systemctl stop fnx.service
docker exec fnx bash -lc "pkill -f '[p]ython -m fnx.server' || true"
sleep 3
docker exec fnx bash -lc 'set -a; source /root/ascend-dev/r109/candidate.env; set +a; export ASCEND_RT_VISIBLE_DEVICES=0; cd /root/ascend-dev/r109; /root/ascend-dev/venv/bin/python delta_test.py'
mkdir -p "$ROOT/r109/short" "$ROOT/r109/long" "$ROOT/r109/short_after"
docker exec -d fnx bash -lc 'set -a; source /root/ascend-dev/r109/candidate.env; set +a; cd /root/ascend-dev/r109; /root/ascend-dev/venv/bin/python -m fnx.server > /root/ascend-dev/logs/r109.server.log 2>&1'
ready=0
for i in $(seq 1 300);do
 sleep 2
 if curl -fs --max-time 3 http://127.0.0.1:8000/health > "$ROOT/r109/health.json";then ready=1;break;fi
 grep -qE 'Traceback|out of memory' "$ROOT/logs/r109.server.log" && break
done
[ "$ready" = 1 ]
docker exec fnx bash -lc 'cd /root/ascend-dev/r109; BENCH_DIR=/root/ascend-dev/r109/short /root/ascend-dev/venv/bin/python r51_payloads.py; BENCH_DIR=/root/ascend-dev/r109/short /root/ascend-dev/venv/bin/python r51_measure.py'
docker exec fnx bash -lc 'cd /root/ascend-dev/r109; TEST_CTX=262144 LONG_OUTPUT=128 LONG_EXPLAIN=1 LONG_CONTINUE=1 RESULT_DIR=/root/ascend-dev/r109/long /root/ascend-dev/venv/bin/python r52_long_client.py'
docker exec fnx bash -lc 'cd /root/ascend-dev/r109; BENCH_DIR=/root/ascend-dev/r109/short_after /root/ascend-dev/venv/bin/python r51_payloads.py; BENCH_DIR=/root/ascend-dev/r109/short_after /root/ascend-dev/venv/bin/python r51_measure.py'
curl -fs --max-time 5 http://127.0.0.1:8000/health > "$ROOT/r109/final-health.json"
echo R109_CANDIDATE_ACCEPT
