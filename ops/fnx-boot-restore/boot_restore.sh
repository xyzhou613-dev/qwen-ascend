#!/bin/bash
# Restore the original container's running intent after boot, not a stopped model.
set -euo pipefail
S=/root/ascend-dev/fnx-boot-restore
exec 9>"$S/restore.lock"
flock -n 9 || { echo RECOVERY_ALREADY_RUNNING;exit 0; }
cd "$S"
python3 - <<'PY'
import json,pathlib,re,subprocess
from reconcile_devices import restore_intended
p=pathlib.Path('/root/ascend-dev/fnx-boot-restore')
d=json.loads(subprocess.check_output(['docker','inspect','bole-model-Flash']))[0]
root=pathlib.Path(subprocess.check_output(['docker','info','-f','{{.DockerRootDir}}'],text=True).strip())
disk=json.load(open(root/'containers'/d['Id']/'config.v2.json'))
intent=restore_intended(d['State']['Running'],d['State'].get('Error',''),disk.get('HasBeenManuallyStopped',False))
if not intent:
 (p/'boot-action').write_text('skip');print('SKIP_MANUALLY_STOPPED_FLASH');raise SystemExit(0)
for row in pathlib.Path('/proc').glob('[0-9]*'):
 try:
  cmd=row.joinpath('cmdline').read_bytes().split(b'\0')
  if b'fnx.server' in cmd or b'fnx.pipeline_worker' in cmd:
   (p/'boot-action').write_text('healthy-check');break
 except (FileNotFoundError,PermissionError,ProcessLookupError):pass
else:(p/'boot-action').write_text('restore')
identity=dict(container_id=d['Id'],docker_root=str(root))
existing=p/'container-identity.json'
if existing.exists():assert json.load(open(existing))==identity,'ORIGINAL_CONTAINER_ID_CHANGED'
else:existing.write_text(json.dumps(identity,indent=2))
PY
if [ "$(cat boot-action)" = skip ];then exit 0;fi
if curl -fsS -m 5 http://127.0.0.1:9002/health > health-check.json;then
 if python3 -c 'import json;h=json.load(open("health-check.json"));assert h["status"]=="ok" and h["worker_alive"]';then
  if ! pgrep -f '(^|/)bash /root/ascend-dev/watchdog.sh' >/dev/null;then systemd-run --unit=fnx-watchdog --collect /bin/bash /root/ascend-dev/watchdog.sh;fi
  echo ALREADY_HEALTHY;exit 0
 fi
fi
if [ "$(cat boot-action)" = healthy-check ];then echo REFUSED_ACTIVE_ENGINE_REPLACEMENT;exit 3;fi
python3 - <<'PY'
import re,subprocess
s=subprocess.check_output(['npu-smi','info'],text=True)
occupied=re.findall(r'\|\s+\d+\s+\d+\s+\|\s+(\d+)\s+\|',s)
assert not occupied,('REFUSED_OCCUPIED_NPU',occupied)
PY
python3 reconcile_devices.py --state "$S"
systemctl stop model-switch.service
trap 'systemctl start docker.service || true;systemctl start model-switch.service || true' EXIT
if python3 -c 'import json;raise SystemExit(0 if json.load(open("device-plan.json"))["changed"] else 1)';then
 test -z "$(docker ps -q)" || { echo DEVICE_REPAIR_REQUIRES_STOPPED_CONTAINERS;exit 4; }
 npu-smi info > before-npu.txt
 # Character nodes are changed only while the daemon is stopped.
 systemctl stop docker.service
 python3 reconcile_devices.py --state "$S" --apply
 systemctl start docker.service
fi
docker start bole-model-Flash
timeout 800 env WD=0 bash /root/ascend-dev/p275.sh > p275.log 2>&1
curl -fsS -m 5 http://127.0.0.1:9002/health > restored-health.json
python3 -c 'import json;h=json.load(open("restored-health.json"));assert h["status"]=="ok" and h["worker_alive"]'
if ! pgrep -f '(^|/)bash /root/ascend-dev/watchdog.sh' >/dev/null;then
 systemd-run --unit=fnx-watchdog --collect /bin/bash /root/ascend-dev/watchdog.sh
fi
systemctl start model-switch.service
echo CANONICAL_FNX_BOOT_RESTORE_COMPLETE
