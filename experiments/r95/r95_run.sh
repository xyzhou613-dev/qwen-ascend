#!/bin/bash
set -euo pipefail
ROOT=/root/ascend-dev
grep -q R94_INDEX_ACCEPT "$ROOT/logs/r94_index.log"
python3 "$ROOT/r95_prepare.py"
cp "$ROOT/r95_state_test.py" "$ROOT/r95/state_test.py"
cp "$ROOT/r94_index_test.py" "$ROOT/r94/index_test.py"
trap 'systemctl start fnx.service' ERR
systemctl stop fnx.service
docker exec fnx bash -lc "pkill -f '[p]ython -m fnx.server' || true"
sleep 3
docker exec fnx bash -lc 'set -a; source /root/ascend-dev/r95/candidate.env; set +a; export ASCEND_RT_VISIBLE_DEVICES=0; cd /root/ascend-dev/r94; /root/ascend-dev/venv/bin/python index_test.py' > "$ROOT/logs/r94_index_final.log" 2>&1
docker exec fnx bash -lc 'set -a; source /root/ascend-dev/r95/candidate.env; set +a; export ASCEND_RT_VISIBLE_DEVICES=0; cd /root/ascend-dev/r95; /root/ascend-dev/venv/bin/python state_test.py'
mkdir -p "$ROOT/r95/short" "$ROOT/r95/long1" "$ROOT/r95/long2" "$ROOT/r95/short_after"
docker exec -d fnx bash -lc 'set -a; source /root/ascend-dev/r95/candidate.env; set +a; cd /root/ascend-dev/r95; /root/ascend-dev/venv/bin/python -m fnx.server > /root/ascend-dev/logs/r95.server.log 2>&1'
ready=0
for i in $(seq 1 300);do
 sleep 2
 if curl -fs --max-time 3 http://127.0.0.1:8000/health > "$ROOT/r95/health.json";then ready=1;break;fi
 grep -qE 'Traceback|out of memory' "$ROOT/logs/r95.server.log" && break
done
[ "$ready" = 1 ]
docker exec fnx bash -lc 'cd /root/ascend-dev/r95; BENCH_DIR=/root/ascend-dev/r95/short /root/ascend-dev/venv/bin/python r51_payloads.py; BENCH_DIR=/root/ascend-dev/r95/short /root/ascend-dev/venv/bin/python r51_measure.py'
for name in long1 long2;do
 docker exec fnx bash -lc "cd /root/ascend-dev/r95; TEST_CTX=262144 LONG_OUTPUT=128 LONG_EXPLAIN=1 LONG_CONTINUE=1 RESULT_DIR=/root/ascend-dev/r95/$name /root/ascend-dev/venv/bin/python r52_long_client.py"
done
docker exec fnx bash -lc 'cd /root/ascend-dev/r95; BENCH_DIR=/root/ascend-dev/r95/short_after /root/ascend-dev/venv/bin/python r51_payloads.py; BENCH_DIR=/root/ascend-dev/r95/short_after /root/ascend-dev/venv/bin/python r51_measure.py'
curl -fs --max-time 5 http://127.0.0.1:8000/health > "$ROOT/r95/final-health.json"
echo R95_CANDIDATE_ACCEPT
