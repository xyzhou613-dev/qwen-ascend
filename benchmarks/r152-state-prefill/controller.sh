#!/bin/bash
set -euo pipefail
cd /models/fnx-diagnostics/r152-state-prefill-20261113
python3 - <<'VERIFY'
import json
from pathlib import Path
p=Path('.')
for d in (0,1):
 r=json.load(open(p/f'all-linear-device{d}.json'));assert r['status']=='passed' and len(r['layers'])==18 and all(x['all_bitwise_equal'] for x in r['layers'])
VERIFY
python3 -c 'import json;f=json.load(open("fixtures.json"));assert len(f)==5 and len(f["natural36000"])==36000'
python3 -c 'import json;assert all(json.load(open(f"component-device{d}.json"))["status"]=="passed" for d in (0,1))'
test -f candidate/fnx/libfnx_state_update.so
test -f baseline/fnx/engine.py
echo $$ > controller.pid
touch lease.active
cleanup() {
 set +e
 if bash "$PWD/rescue.sh" && test ! -f lease.active;then systemctl stop fnx-r152-rescue.timer;else echo RESTORE_PENDING_TIMER_RETAINED >&2;fi
}
trap cleanup EXIT
trap 'exit 143' TERM INT
systemd-run --unit=fnx-r152-rescue --on-active=40min /bin/bash "$PWD/rescue.sh"
systemctl stop model-switch.service
mkdir -p /run/systemd/system/model-switch.service.d
printf '[Unit]\nRefuseManualStart=yes\n' > /run/systemd/system/model-switch.service.d/r152-exclusive.conf
systemctl daemon-reload
systemctl stop fnx-r151-watchdog.service 2>/dev/null || true
if pgrep -f '(^|/)bash /root/ascend-dev/watchdog.sh' > watchdog.pids;then while read -r pid;do kill -TERM "$pid";done < watchdog.pids;fi
docker exec bole-model-Flash /root/ascend-dev/venv/bin/python "$PWD/snapshot.py"
docker exec bole-model-Flash /root/ascend-dev/venv/bin/python "$PWD/stop_owned.py"
for i in $(seq 1 40);do test -z "$(docker exec bole-model-Flash pgrep -f '[p]ython -m fnx.server|[p]ython -m fnx.pipeline_worker' || true)" && break;sleep 1;done
test -z "$(docker exec bole-model-Flash pgrep -f '[p]ython -m fnx.server|[p]ython -m fnx.pipeline_worker' || true)"
timeout 400 docker exec bole-model-Flash timeout -k 5 380 /root/ascend-dev/venv/bin/python "$PWD/launch.py" layer_probe.py > layer-probe.log 2>&1
grep -q REAL_LAYER_COMPLETE layer-probe.log
for phase in baseline candidate;do
 docker exec bole-model-Flash /root/ascend-dev/venv/bin/python "$PWD/stop_owned.py"
 date +%s > "${phase}-start-epoch.txt"
 docker exec bole-model-Flash /root/ascend-dev/venv/bin/python "$PWD/launch.py" --server "$phase"
 ready=0
 for i in $(seq 1 360);do sleep 2;if curl -fsS -m 2 http://127.0.0.1:19002/health > "${phase}-health.json";then ready=1;break;fi;done
 test "$ready" = 1
 PHASE="$phase" python3 - <<'PY'
import json,os,time
from pathlib import Path
p=Path('.');phase=os.environ['PHASE'];h=json.load(open(p/f'{phase}-health.json'))
assert h['worker_alive'] and h['diagnostic_variant']==phase and h['state_update']==(phase=='candidate')
r=dict(phase=phase,start_to_http_ready_s=time.time()-int((p/f'{phase}-start-epoch.txt').read_text()))
(p/f'{phase}-startup.json').write_text(json.dumps(r,indent=2));print('HTTP_READY',r,flush=True)
PY
 python3 quality_client.py "$phase"
 python3 http_bench.py "$phase"
done
python3 evaluate.py
echo RESTART_AB_COMPLETE
