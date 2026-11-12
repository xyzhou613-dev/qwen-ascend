#!/bin/bash
set -euo pipefail
cd /models/fnx-diagnostics/r144-raw-norm-20261112
grep -q CANDIDATE_TESTS_COMPLETE controller.log
if [ "${R144_HANDOFF:-0}" != 1 ];then
 grep -q R143_CANONICAL_AND_MANAGER_RESTORED controller.log
 test ! -f lease.active
fi
echo $$ > deployment.pid
touch deployment.active
cleanup() {
 set +e
 bash "$PWD/deployment_rescue.sh"
 systemctl stop fnx-r144-deploy-rescue.timer
}
trap cleanup EXIT
trap 'exit 143' TERM INT
systemd-run --unit=fnx-r144-deploy-rescue --on-active=22min /bin/bash "$PWD/deployment_rescue.sh"
if [ "${R144_HANDOFF:-0}" = 1 ];then systemctl stop fnx-r144-rescue4.timer;fi
systemctl stop model-switch.service
mkdir -p /run/systemd/system/model-switch.service.d
printf '[Unit]\nRefuseManualStart=yes\n' > /run/systemd/system/model-switch.service.d/r144-exclusive.conf
systemctl daemon-reload
if pgrep -f '^bash /root/ascend-dev/watchdog.sh$' > watchdog.pids;then while read -r pid;do kill -TERM "$pid";done < watchdog.pids;fi
docker exec bole-model-Flash /root/ascend-dev/venv/bin/python "$PWD/rescue_stop.py"
python3 deployment_files.py install
timeout 780 env WD=0 bash /root/ascend-dev/p275.sh > canonical-start.log 2>&1
curl -fs -m 5 http://127.0.0.1:9002/health > production-health.json
python3 - <<'PY'
import json
h=json.load(open('production-health.json'));assert h['service_revision']=='r144' and h['worker_alive'] and h['norm_post'] and h['delta_reuse']
PY
mkdir -p production8k production256
cp baseline8k/payload*.json production8k/
docker exec -e BENCH_DIR="$PWD/production8k" bole-model-Flash /root/ascend-dev/venv/bin/python "$PWD/r51_measure.py"
docker exec -e TEST_CTX=262144 -e LONG_OUTPUT=128 -e LONG_EXPLAIN=1 -e RESULT_DIR="$PWD/production256" bole-model-Flash /root/ascend-dev/venv/bin/python "$PWD/r52_long_client.py"
python3 - <<'PY'
import json,statistics
b=json.load(open('baseline8k/results.json'));c=json.load(open('production8k/results.json'))
m=lambda r,k:statistics.median(x[k] for x in r)
assert m(c,'decode_tpot_ms')<m(b,'decode_tpot_ms')*.995
assert m(c,'ttft_s')<m(b,'ttft_s')*1.01
b=json.load(open('baseline256/result_262144.json'));c=json.load(open('production256/result_262144.json'))
assert c['stats']['tpot_ms']<b['stats']['tpot_ms']*.995
assert c['ttft_s']<b['ttft_s']*1.01
print('PRODUCTION_ACCEPT',flush=True)
PY
rm -f /run/systemd/system/model-switch.service.d/r144-exclusive.conf
systemctl daemon-reload
systemctl start model-switch.service
bash "$PWD/restore_watchdog.sh"
rm -f deployment.active
echo PRODUCTION_AND_MANAGER_COMPLETE
