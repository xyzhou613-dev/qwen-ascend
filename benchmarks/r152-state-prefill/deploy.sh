#!/bin/bash
set -euo pipefail
cd /models/fnx-diagnostics/r152-state-prefill-20261113
python3 -c 'import json;assert json.load(open("ab-summary.json"))["accepted"]'
grep -q CANONICAL_R151_MANAGER_RESTORED controller.log
test ! -f lease.active
test -f deployment-backup/manifest.json
echo $$ > deployment.pid
touch deployment.active
cleanup() {
 set +e
 if bash "$PWD/deploy_rescue.sh" && test ! -f deployment.active;then systemctl stop fnx-r152-deploy-rescue.timer;else echo DEPLOY_RESTORE_PENDING_TIMER_RETAINED >&2;fi
}
trap cleanup EXIT
trap 'exit 143' TERM INT
systemd-run --unit=fnx-r152-deploy-rescue --on-active=25min /bin/bash "$PWD/deploy_rescue.sh"
systemctl stop model-switch.service
mkdir -p /run/systemd/system/model-switch.service.d
printf '[Unit]\nRefuseManualStart=yes\n' > /run/systemd/system/model-switch.service.d/r152-exclusive.conf
systemctl daemon-reload
systemctl stop fnx-r152-watchdog.service 2>/dev/null || true
if pgrep -f '(^|/)bash /root/ascend-dev/watchdog.sh' > watchdog.pids;then while read -r pid;do kill -TERM "$pid";done < watchdog.pids;fi
docker exec bole-model-Flash /root/ascend-dev/venv/bin/python "$PWD/deploy_stop.py"
python3 deployment_files.py install
timeout 780 env WD=0 bash /root/ascend-dev/p275.sh > production-start.log 2>&1
curl -fsS -m 5 http://127.0.0.1:9002/health > production-health.json
python3 -c 'import json;h=json.load(open("production-health.json"));assert h["service_revision"]=="r152" and h["worker_alive"] and h["state_update"] and h["norm_post"] and h["delta_reuse"] and h["ctx"]==262144'
BENCH_URL=http://127.0.0.1:9002 python3 http_bench.py production
python3 - <<'PY'
import json
from pathlib import Path
b=json.load(open('baseline/8k.json'));c=json.load(open('production/8k.json'));assert len(b)==len(c)==3 and all(x['answer']==y['answer'] for x,y in zip(b,c))
for f in ('256k.json','256k-second.json','256k-third.json','cached256k.json'):assert json.load(open('baseline/'+f))['answer']==json.load(open('production/'+f))['answer']
a=json.load(open('baseline/summary.json'));b=json.load(open('production/summary.json'))
reduction=100*(1-b['ttft_256k_s']/a['ttft_256k_s']);runs=[100*(1-y/x) for x,y in zip(a['ttft_256k_runs_s'],b['ttft_256k_runs_s'])]
assert len(runs)==3 and reduction>.5 and all(v>0 for v in runs),('production TTFT gate failed',reduction,runs)
for k in ('median_8k_ttft_s','median_8k_tpot_ms','tpot_256k_ms'):assert b[k]<=a[k]*1.01,('production regression',k,a[k],b[k])
Path('production-verification.json').write_text(json.dumps(dict(answers_exact=True,ttft_256k_reduction_pct=reduction,individual_long_reductions_pct=runs,accepted=True),indent=2))
print('PRODUCTION_ANSWERS_EXACT_AND_TTFT_GAIN_REPRODUCED',reduction,runs,flush=True)
PY
rm -f /run/systemd/system/model-switch.service.d/r152-exclusive.conf
systemctl daemon-reload
systemctl start model-switch.service
systemd-run --unit=fnx-r152-watchdog --collect /bin/bash /root/ascend-dev/watchdog.sh
rm -f deployment.active
echo R152_PRODUCTION_VALIDATED_MANAGER_RESTORED
