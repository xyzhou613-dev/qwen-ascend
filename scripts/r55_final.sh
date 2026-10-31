#!/bin/bash
set -euo pipefail
systemctl is-active fnx.service
docker exec fnx bash -lc 'cd /root/ascend-dev/r55; TEST_CTX=262144 LONG_OUTPUT=128 LONG_EXPLAIN=1 RESULT_DIR=/root/ascend-dev/r55/production /root/ascend-dev/venv/bin/python r52_long_client.py' > /root/ascend-dev/logs/r55_production256.log 2>&1
python3 - <<'PY'
import json
from pathlib import Path
r=json.loads(Path('/root/ascend-dev/r55/production/result_262144.json').read_text())
assert r['stats']['n_prefill']==262012 and r['recall_hits']==3
assert r['stats']['tokens']>=120
assert r['ttft_s']<=120,(r['ttft_s'],'TTFT goal')
assert r['stats']['tpot_ms']<=1000/15,(r['stats']['tpot_ms'],'decode goal')
print('PRODUCTION_TARGET_ACCEPT',r['ttft_s'],1000/r['stats']['tpot_ms'])
PY
docker exec fnx bash -lc 'cd /root/ascend-dev/r55; BENCH_DIR=/root/ascend-dev/r55/production8k /root/ascend-dev/venv/bin/python r51_payloads.py; BENCH_DIR=/root/ascend-dev/r55/production8k /root/ascend-dev/venv/bin/python r51_measure.py' > /root/ascend-dev/logs/r55_production8k.log 2>&1
systemctl is-active fnx.service
curl -fsS --max-time 10 http://127.0.0.1:8000/health
echo
echo FINAL_PRODUCTION_ACCEPT
