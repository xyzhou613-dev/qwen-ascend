#!/bin/bash
set -euo pipefail
cd /models/fnx-diagnostics/r143-index-post-20261112
echo $$ > controller.pid
touch lease.active
cleanup() {
 set +e
 bash "$PWD/rescue.sh"
 systemctl stop fnx-r143-stage-a-rescue.timer
}
trap cleanup EXIT
trap 'exit 143' TERM INT
systemd-run --unit=fnx-r143-stage-a-rescue --on-active=18min /bin/bash "$PWD/rescue.sh"
grep -q E2E_COMPLETE baseline.log
grep -q 'LONG_ACCEPT 262144' baseline256.log
python3 - <<'PY'
import json
a=json.load(open('probe.json'));assert all(r['exact'] for r in a if r.get('mode')==1 and r['kind']!='timing')
assert json.load(open('selection.json'))['mode']==1
PY
systemctl stop model-switch.service
mkdir -p /run/systemd/system/model-switch.service.d
printf '[Unit]\nRefuseManualStart=yes\n' > /run/systemd/system/model-switch.service.d/r143-exclusive.conf
systemctl daemon-reload
if pgrep -f '^bash /root/ascend-dev/watchdog.sh$' > watchdog.pids;then
 touch watchdog.was-running
 while read -r pid;do kill -TERM "$pid";done < watchdog.pids
fi
docker exec bole-model-Flash /root/ascend-dev/venv/bin/python "$PWD/snapshot.py"
docker exec bole-model-Flash /root/ascend-dev/venv/bin/python "$PWD/stop_owned.py" original
for i in $(seq 1 30);do
 if ! docker exec bole-model-Flash pgrep -f '[p]ython -m fnx.server|[p]ython -m fnx.pipeline_worker' >/dev/null;then break;fi
 sleep 1
done
test -z "$(docker exec bole-model-Flash pgrep -f '[p]ython -m fnx.server|[p]ython -m fnx.pipeline_worker' || true)"
timeout 300 docker exec bole-model-Flash /root/ascend-dev/venv/bin/python "$PWD/launch.py" full_layer_probe.py > full-layer.log 2>&1
grep -q REAL_FULL_LAYER_ACCEPT full-layer.log
mkdir -p candidate8k candidate256
docker exec bole-model-Flash /root/ascend-dev/venv/bin/python "$PWD/launch.py"
ready=0
for i in $(seq 1 300);do
 sleep 2
 if curl -fs --max-time 2 http://127.0.0.1:9002/health > candidate-health.json;then ready=1;break;fi
done
test "$ready" = 1
python3 - <<'PY'
import json
h=json.load(open('candidate-health.json'));assert h['service_revision']=='r143' and h['index_post'] and h['worker_alive']
PY
python3 quality_client.py
cp baseline/payload*.json candidate8k/
docker exec -e BENCH_DIR="$PWD/candidate8k" bole-model-Flash /root/ascend-dev/venv/bin/python "$PWD/launch.py" r51_measure.py
python3 - <<'PY'
from pathlib import Path
import json,statistics
for i in (1,2,3):assert Path(f'baseline/answer{i}.txt').read_bytes()==Path(f'candidate8k/answer{i}.txt').read_bytes(),i
m=lambda p,k:statistics.median(r[k] for r in json.load(open(p+'/results.json')))
assert m('candidate8k','decode_tpot_ms')<=m('baseline','decode_tpot_ms')*1.01
assert m('candidate8k','ttft_s')<=m('baseline','ttft_s')*1.01
print('THREE_8K_EXACT_AND_NO_REGRESSION',flush=True)
PY
docker exec -e TEST_CTX=262144 -e LONG_OUTPUT=128 -e LONG_EXPLAIN=1 -e RESULT_DIR="$PWD/candidate256" bole-model-Flash /root/ascend-dev/venv/bin/python "$PWD/launch.py" r52_long_client.py
python3 - <<'PY'
import json
b=json.load(open('baseline256/result_262144.json'));c=json.load(open('candidate256/result_262144.json'))
assert c['ttft_s']<=b['ttft_s']*1.01 and c['stats']['tpot_ms']<=b['stats']['tpot_ms']*1.01
assert c['ttft_s']<b['ttft_s']*.99 or c['stats']['tpot_ms']<b['stats']['tpot_ms']*.995
print('LONG_CONTEXT_SPEED_ACCEPT',flush=True)
PY
echo CANDIDATE_ACCEPT_COMPLETE
