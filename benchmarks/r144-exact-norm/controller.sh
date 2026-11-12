#!/bin/bash
set -euo pipefail
cd /models/fnx-diagnostics/r144-raw-norm-20261112
test -f baseline-quality.json
echo $$ > controller.pid
touch lease.active
cleanup() {
 set +e
 bash "$PWD/rescue.sh"
 systemctl stop fnx-r144-rescue4.timer
}
trap cleanup EXIT
trap 'exit 143' TERM INT
systemd-run --unit=fnx-r144-rescue4 --on-active=25min /bin/bash "$PWD/rescue.sh"
systemctl stop model-switch.service
mkdir -p /run/systemd/system/model-switch.service.d
printf '[Unit]\nRefuseManualStart=yes\n' > /run/systemd/system/model-switch.service.d/r144-exclusive.conf
systemctl daemon-reload
if pgrep -f '^bash /root/ascend-dev/watchdog.sh$' > watchdog.pids;then while read -r pid;do kill -TERM "$pid";done < watchdog.pids;fi
docker exec bole-model-Flash /root/ascend-dev/venv/bin/python "$PWD/snapshot.py"
docker exec bole-model-Flash /root/ascend-dev/venv/bin/python "$PWD/stop_owned.py" original
sleep 4
timeout 300 docker exec bole-model-Flash /root/ascend-dev/venv/bin/python "$PWD/launch.py" full_layer_probe.py > full-layer.log 2>&1
grep -q FULL_LAYER_COMPLETE full-layer.log
docker exec -e FNX_NORM_POST=1 -e FNX_DELTA_REUSE=1 bole-model-Flash /root/ascend-dev/venv/bin/python "$PWD/launch.py"
ready=0
for i in $(seq 1 300);do sleep 2;if curl -fs --max-time 2 http://127.0.0.1:9002/health > diagnostic-health.json;then ready=1;break;fi;done
test "$ready" = 1
python3 quality_client.py candidate
python3 - <<'PY'
import json
b=json.load(open('baseline-quality.json'));c=json.load(open('candidate-quality.json'))
for name,x in b.items():
 y=c[name];assert y['token_ids_sha256']==x['token_ids_sha256']
 assert y['nll_tokens']==x['nll_tokens'] and y['argmax']==x['argmax'],name
print('MULTITEXT_QUALITY_GATE_PASS',flush=True)
PY
mkdir -p candidate8k candidate256
cp baseline8k/payload*.json candidate8k/
docker exec -e BENCH_DIR="$PWD/candidate8k" bole-model-Flash /root/ascend-dev/venv/bin/python "$PWD/launch.py" r51_measure.py
docker exec -e TEST_CTX=262144 -e LONG_OUTPUT=128 -e LONG_EXPLAIN=1 -e RESULT_DIR="$PWD/candidate256" bole-model-Flash /root/ascend-dev/venv/bin/python "$PWD/launch.py" r52_long_client.py
python3 - <<'PY'
import json,statistics
from pathlib import Path
b=json.load(open('baseline8k/results.json'));a=json.load(open('baseline-after8k/results.json'));c=json.load(open('candidate8k/results.json'))
m=lambda r,k:statistics.median(x[k] for x in r)
assert m(c,'decode_tpot_ms')<min(m(b,'decode_tpot_ms'),m(a,'decode_tpot_ms'))*.995
assert m(c,'ttft_s')<max(m(b,'ttft_s'),m(a,'ttft_s'))*1.01
print('ANSWERS_IDENTICAL',[Path(f'baseline8k/answer{i}.txt').read_bytes()==Path(f'candidate8k/answer{i}.txt').read_bytes() for i in (1,2,3)],flush=True)
print('THREE_FRESH_8K_FASTER',flush=True)
PY
echo CANDIDATE_TESTS_COMPLETE
