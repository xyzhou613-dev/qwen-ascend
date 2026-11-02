#!/bin/bash
set -euo pipefail
ROOT=/root/ascend-dev
grep -q R95_PRODUCTION_ACCEPT "$ROOT/logs/r95_production.log"
# The HTTP benchmark has finished. Keep the healthy service resident and idle;
# these bounded probes need less than 2 GiB additional HBM, not a second model.
cp "$ROOT/r95/state_test.py" "$ROOT/r95/state_test_full_initial.py"
cp "$ROOT/r95_state_test.py" "$ROOT/r95/state_test.py"
cp "$ROOT/r97_rank_probe.py" "$ROOT/r95/rank_probe.py"
docker exec fnx bash -lc 'set -a; source /root/ascend-dev/r95/candidate.env; set +a; export ASCEND_RT_VISIBLE_DEVICES=0 FNX_STATE_TEST_CTX=8192 FNX_STATE_TEST_OUTPUT=/root/ascend-dev/r95/state-all-bits.json; cd /root/ascend-dev/r95; /root/ascend-dev/venv/bin/python state_test.py' > "$ROOT/logs/r95_all_bits.log" 2>&1
docker exec fnx bash -lc 'cd /root/ascend-dev/r95; ASCEND_RT_VISIBLE_DEVICES=0 /root/ascend-dev/venv/bin/python rank_probe.py'
curl -fs --max-time 5 http://127.0.0.1:8000/health > "$ROOT/r95/post-probe-health.json"
echo R97_PROBE_ACCEPT
