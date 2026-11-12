#!/bin/bash
set -euo pipefail
cd /models/fnx-diagnostics/r145-hardware-profile-20261112
grep -q METRIC_PROBE_COMPLETE metric-probe.log
test -f fnx/libfnx_norm_post.so
test -f payload256k.json
test -f runs-8k.json
echo $$ > controller.pid
touch lease.active
cleanup() {
 set +e
 bash "$PWD/rescue.sh"
 systemctl stop fnx-r145-rescue2.timer
}
trap cleanup EXIT
trap 'exit 143' TERM INT
systemd-run --unit=fnx-r145-rescue2 --on-active=25min /bin/bash "$PWD/rescue.sh"
systemctl stop fnx-profile-watchdog.service 2>/dev/null || true
systemctl stop model-switch.service
mkdir -p /run/systemd/system/model-switch.service.d
printf '[Unit]\nRefuseManualStart=yes\n' > /run/systemd/system/model-switch.service.d/r145-profile.conf
systemctl daemon-reload
if pgrep -f '(^|/)bash /root/ascend-dev/watchdog.sh' > watchdog.pids;then while read -r pid;do kill -TERM "$pid";done < watchdog.pids;fi
docker exec bole-model-Flash /root/ascend-dev/venv/bin/python "$PWD/snapshot.py"
docker exec bole-model-Flash /root/ascend-dev/venv/bin/python "$PWD/stop_owned.py" original
for i in $(seq 1 40);do test -z "$(docker exec bole-model-Flash pgrep -f '[p]ython -m fnx.server|[p]ython -m fnx.pipeline_worker' || true)" && break;sleep 1;done
test -z "$(docker exec bole-model-Flash pgrep -f '[p]ython -m fnx.server|[p]ython -m fnx.pipeline_worker' || true)"
timeout 1200 docker exec -e R145_CONTEXTS=256k bole-model-Flash /root/ascend-dev/venv/bin/python "$PWD/launch.py" profile_model.py > profile.log 2>&1
grep -q R145_PROFILE_COMPLETE profile.log
echo PROFILING_CAPTURE_COMPLETE
