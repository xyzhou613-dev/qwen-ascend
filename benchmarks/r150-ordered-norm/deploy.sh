#!/bin/bash
set -euo pipefail
cd /models/fnx-diagnostics/r150-ordered-norm-20261112
python3 -c 'import json;assert json.load(open("ab-summary.json"))["accepted"]'
grep -q CANONICAL_R144_MANAGER_RESTORED controller.log
test ! -f lease.active
test -f deployment-backup/manifest.json
echo $$ > deployment.pid
touch deployment.active
cleanup() {
 set +e
 if bash "$PWD/deploy_rescue.sh" && test ! -f deployment.active;then systemctl stop fnx-r150-deploy-rescue.timer;else echo DEPLOY_RESTORE_PENDING_TIMER_RETAINED >&2;fi
}
trap cleanup EXIT
trap 'exit 143' TERM INT
systemd-run --unit=fnx-r150-deploy-rescue --on-active=25min /bin/bash "$PWD/deploy_rescue.sh"
systemctl stop model-switch.service
mkdir -p /run/systemd/system/model-switch.service.d
printf '[Unit]\nRefuseManualStart=yes\n' > /run/systemd/system/model-switch.service.d/r150-exclusive.conf
systemctl daemon-reload
systemctl stop fnx-r150-watchdog.service 2>/dev/null || true
if pgrep -f '(^|/)bash /root/ascend-dev/watchdog.sh' > watchdog.pids;then while read -r pid;do kill -TERM "$pid";done < watchdog.pids;fi
docker exec bole-model-Flash /root/ascend-dev/venv/bin/python "$PWD/deploy_stop.py"
python3 deployment_files.py install
timeout 780 env WD=0 bash /root/ascend-dev/p275.sh > production-start.log 2>&1
curl -fsS -m 5 http://127.0.0.1:9002/health > production-health.json
python3 -c 'import json;h=json.load(open("production-health.json"));assert h["service_revision"]=="r150" and h["worker_alive"] and h["norm_ordered"] and h["norm_post"] and h["delta_reuse"] and h["ctx"]==262144'
BENCH_URL=http://127.0.0.1:9002 python3 http_bench.py production
python3 - <<'PY'
import json
b=json.load(open('baseline/8k.json'));p=json.load(open('production/8k.json'))
assert all(x['answer']==y['answer'] for x,y in zip(b,p))
assert json.load(open('baseline/256k.json'))['answer']==json.load(open('production/256k.json'))['answer']
assert json.load(open('baseline/cached256k.json'))['answer']==json.load(open('production/cached256k.json'))['answer']
a=json.load(open('baseline/summary.json'));b=json.load(open('production/summary.json'))
gains={k:100*(a[k]/b[k]-1) for k in ('median_8k_tpot_ms','tpot_256k_ms')}
assert all(v>.5 for v in gains.values()),('production performance gate failed',gains)
Path=__import__('pathlib').Path
Path('production-verification.json').write_text(json.dumps(dict(answers_exact=True,speedups_pct=gains,accepted=True),indent=2))
print('PRODUCTION_ANSWERS_EXACT_AND_GAIN_REPRODUCED',gains,flush=True)
PY
rm -f /run/systemd/system/model-switch.service.d/r150-exclusive.conf
systemctl daemon-reload
systemctl start model-switch.service
systemd-run --unit=fnx-r150-watchdog --collect /bin/bash /root/ascend-dev/watchdog.sh
rm -f deployment.active
echo R150_PRODUCTION_VALIDATED_MANAGER_RESTORED
