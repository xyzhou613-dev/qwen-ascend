#!/bin/bash
set -euo pipefail
cd /models/fnx-diagnostics/r143-index-post-20261112
echo $$ > controller.pid
touch lease.active
success=0
cleanup() {
 set +e
 if [ "$success" != 1 ];then
  bash "$PWD/rescue.sh"
 else
  rm -f /run/systemd/system/model-switch.service.d/r143-deployment.conf
  systemctl daemon-reload
  systemctl start model-switch.service
  if [ -f watchdog.was-running ] && ! pgrep -f '^bash /root/ascend-dev/watchdog.sh$' >/dev/null;then
   nohup bash /root/ascend-dev/watchdog.sh > /root/ascend-dev/logs/watchdog.log 2>&1 < /dev/null &
  fi
  rm -f lease.active
  echo MANAGER_RESTORED
 fi
 systemctl stop fnx-r143-stage-b-rescue.timer
}
trap cleanup EXIT
trap 'exit 143' TERM INT
systemd-run --unit=fnx-r143-stage-b-rescue --on-active=15min /bin/bash "$PWD/rescue.sh"
grep -q R142_AND_MANAGER_RESTORED controller.log
test -s CODE_COMMITTED
python3 r143-enable.py /root/ascend-dev/p275.sh --check
systemctl stop model-switch.service
mkdir -p /run/systemd/system/model-switch.service.d
printf '[Unit]\nRefuseManualStart=yes\n' > /run/systemd/system/model-switch.service.d/r143-deployment.conf
systemctl daemon-reload
if pgrep -f '^bash /root/ascend-dev/watchdog.sh$' > watchdog.pids;then
 touch watchdog.was-running
 while read -r pid;do kill -TERM "$pid";done < watchdog.pids
fi
docker exec bole-model-Flash /root/ascend-dev/venv/bin/python "$PWD/snapshot.py"
mkdir -p baseline-after baseline-after256 production8k production256
cp baseline/payload*.json baseline-after/
docker exec -e BENCH_DIR="$PWD/baseline-after" bole-model-Flash /root/ascend-dev/venv/bin/python "$PWD/launch.py" r51_measure.py
docker exec -e TEST_CTX=262144 -e LONG_OUTPUT=128 -e LONG_EXPLAIN=1 -e RESULT_DIR="$PWD/baseline-after256" bole-model-Flash /root/ascend-dev/venv/bin/python "$PWD/launch.py" r52_long_client.py
docker exec bole-model-Flash /root/ascend-dev/venv/bin/python "$PWD/stop_owned.py" original
for i in $(seq 1 30);do
 if ! docker exec bole-model-Flash pgrep -f '[p]ython -m fnx.server|[p]ython -m fnx.pipeline_worker' >/dev/null;then break;fi
 sleep 1
done
test -z "$(docker exec bole-model-Flash pgrep -f '[p]ython -m fnx.server|[p]ython -m fnx.pipeline_worker' || true)"
python3 deploy.py
timeout 720 env WD=0 bash /root/ascend-dev/p275.sh > canonical-start.log 2>&1
docker exec bole-model-Flash /root/ascend-dev/venv/bin/python "$PWD/production_pid.py"
curl -fs --max-time 5 http://127.0.0.1:9002/health > production-health.json
python3 - <<'PY'
import json
h=json.load(open('production-health.json'));assert h['service_revision']=='r143' and h['index_post'] and h['ctx']==262144 and h['worker_alive']
PY
cp baseline/payload*.json production8k/
docker exec -e BENCH_DIR="$PWD/production8k" bole-model-Flash /root/ascend-dev/venv/bin/python "$PWD/launch.py" r51_measure.py
python3 - <<'PY'
from pathlib import Path
import json,statistics
for i in (1,2,3):assert Path(f'baseline/answer{i}.txt').read_bytes()==Path(f'production8k/answer{i}.txt').read_bytes(),i
m=lambda d,k:statistics.median(r[k] for r in json.load(open(d+'/results.json')))
assert m('production8k','ttft_s')<=m('baseline-after','ttft_s')*1.01
assert m('production8k','decode_tpot_ms')<=m('baseline-after','decode_tpot_ms')*1.01
print('PRODUCTION_8K_EXACT_AND_NO_REGRESSION',flush=True)
PY
docker exec -e TEST_CTX=262144 -e LONG_OUTPUT=128 -e LONG_EXPLAIN=1 -e RESULT_DIR="$PWD/production256" bole-model-Flash /root/ascend-dev/venv/bin/python "$PWD/launch.py" r52_long_client.py
python3 - <<'PY'
import json
b=json.load(open('baseline-after256/result_262144.json'));c=json.load(open('production256/result_262144.json'))
assert c['ttft_s']<=b['ttft_s']*1.01 and c['stats']['tpot_ms']<=b['stats']['tpot_ms']*1.01
assert c['ttft_s']<b['ttft_s']*.99 or c['stats']['tpot_ms']<b['stats']['tpot_ms']*.995
print('PRODUCTION_LONG_SPEED_ACCEPT',flush=True)
PY
success=1
echo R143_PRODUCTION_ACCEPT_COMPLETE
