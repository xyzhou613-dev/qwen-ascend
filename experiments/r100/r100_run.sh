#!/bin/bash
set -euo pipefail
ROOT=/root/ascend-dev
grep -q R95_COMMITTED "$ROOT/logs/r95_finish.log"
grep -q R99_RESTORE_ACCEPT "$ROOT/logs/r99_restore_probe.log"
python3 "$ROOT/r100_prepare.py"
cp "$ROOT/r100_moe_test.py" "$ROOT/r100/moe_test.py"
trap 'systemctl start fnx.service' ERR
systemctl stop fnx.service
docker exec fnx bash -lc "pkill -f '[p]ython -m fnx.server' || true"
sleep 3
docker exec fnx bash -lc 'set -a; source /root/ascend-dev/r100/candidate.env; set +a; export ASCEND_RT_VISIBLE_DEVICES=0; cd /root/ascend-dev/r100; /root/ascend-dev/venv/bin/python moe_test.py'
mkdir -p "$ROOT/r100/short" "$ROOT/r100/long" "$ROOT/r100/short_after"
docker exec -d fnx bash -lc 'set -a; source /root/ascend-dev/r100/candidate.env; set +a; cd /root/ascend-dev/r100; /root/ascend-dev/venv/bin/python -m fnx.server > /root/ascend-dev/logs/r100.server.log 2>&1'
ready=0
for i in $(seq 1 300);do
 sleep 2
 if curl -fs --max-time 3 http://127.0.0.1:8000/health > "$ROOT/r100/health.json";then ready=1;break;fi
 grep -qE 'Traceback|out of memory' "$ROOT/logs/r100.server.log" && break
done
[ "$ready" = 1 ]
docker exec fnx bash -lc 'cd /root/ascend-dev/r100; BENCH_DIR=/root/ascend-dev/r100/short /root/ascend-dev/venv/bin/python r51_payloads.py; BENCH_DIR=/root/ascend-dev/r100/short /root/ascend-dev/venv/bin/python r51_measure.py'
docker exec fnx bash -lc 'cd /root/ascend-dev/r100; TEST_CTX=262144 LONG_OUTPUT=128 LONG_EXPLAIN=1 LONG_CONTINUE=1 RESULT_DIR=/root/ascend-dev/r100/long /root/ascend-dev/venv/bin/python r52_long_client.py'
docker exec fnx bash -lc 'cd /root/ascend-dev/r100; BENCH_DIR=/root/ascend-dev/r100/short_after /root/ascend-dev/venv/bin/python r51_payloads.py; BENCH_DIR=/root/ascend-dev/r100/short_after /root/ascend-dev/venv/bin/python r51_measure.py'
curl -fs --max-time 5 http://127.0.0.1:8000/health > "$ROOT/r100/final-health.json"
echo R100_CANDIDATE_ACCEPT
