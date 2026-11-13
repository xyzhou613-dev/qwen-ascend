#!/bin/bash
set -euo pipefail
P=/models/fnx-diagnostics/r156-moe-kmajor-20261113
cd "$P"
test -z "$(docker ps -q)" || { echo UNRELATED_RUNNING_CONTAINER_REFUSED;exit 2; }
test -c /dev/davinci0 && test -c /dev/davinci1
test ! -e /dev/davinci3 && test ! -e /dev/davinci5
mkdir -p private-recovery;chmod 700 private-recovery
docker inspect -f '{{.Id}}' bole-model-Flash > private-recovery/container-id
docker info -f '{{.DockerRootDir}}' > private-recovery/docker-root
python3 - <<'PY'
import json,pathlib,re,shutil
p=pathlib.Path('/models/fnx-diagnostics/r156-moe-kmajor-20261113/private-recovery')
cid=(p/'container-id').read_text().strip();root=pathlib.Path((p/'docker-root').read_text().strip()).resolve()
assert re.fullmatch('[0-9a-f]{64}',cid) and root.is_absolute()
target=root/'containers'/cid/'hostconfig.json';assert target.is_file()
old=json.loads(target.read_text());devices=old['Devices']
assert {d['PathOnHost'] for d in devices if re.fullmatch('/dev/davinci[0-9]+',d['PathOnHost'])}=={'/dev/davinci3','/dev/davinci5'}
shutil.copy2(target,p/'hostconfig-before.json');(p/'hostconfig-before.json').chmod(0o600)
(p/'target-path').write_text(str(target))
PY
systemctl stop model-switch.service
ensure_daemons(){ systemctl start docker.service || true;systemctl start model-switch.service || true; }
trap ensure_daemons EXIT
systemctl stop docker.service
python3 - <<'PY'
import json,os,pathlib,tempfile
p=pathlib.Path('/models/fnx-diagnostics/r156-moe-kmajor-20261113/private-recovery')
target=pathlib.Path((p/'target-path').read_text());old=json.loads((p/'hostconfig-before.json').read_text());before=[dict(d) for d in old['Devices']]
mapping={'/dev/davinci3':'/dev/davinci1','/dev/davinci5':'/dev/davinci0'}
for d in old['Devices']:
 if d['PathOnHost'] in mapping:
  assert d['PathInContainer']==d['PathOnHost'];d['PathOnHost']=d['PathInContainer']=mapping[d['PathOnHost']]
st=target.stat();temp=target.with_suffix('.r156.tmp');temp.write_text(json.dumps(old,separators=(',',':')))
os.chmod(temp,st.st_mode);os.chown(temp,st.st_uid,st.st_gid);os.replace(temp,target)
(p.parent/'recovery-device-change.json').write_text(json.dumps(dict(reason='host reboot changed available device node indices',boot_id=pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip(),container_recreated=False,before=before,after=old['Devices']),indent=2))
PY
if ! systemctl start docker.service;then
 python3 - <<'PY'
import pathlib,shutil
p=pathlib.Path('/models/fnx-diagnostics/r156-moe-kmajor-20261113/private-recovery');shutil.copy2(p/'hostconfig-before.json',pathlib.Path((p/'target-path').read_text()))
PY
 echo DOCKER_START_FAILED_CONFIG_REVERTED;exit 3
fi
docker start bole-model-Flash
docker exec bole-model-Flash npu-smi info > recovered-container-npu.txt
timeout 800 env WD=0 bash /root/ascend-dev/p275.sh > recover-p275.log 2>&1
curl -fsS -m 5 http://127.0.0.1:9002/health > recovered-health.json
python3 - <<'PY'
import hashlib,json,pathlib
p=pathlib.Path('/models/fnx-diagnostics/r156-moe-kmajor-20261113');h=json.load(open(p/'recovered-health.json'))
assert h['service_revision']=='r152' and h['worker_alive'] and h['ctx']==262144
ref=json.load(open(p/'production-before-hashes.json'))
assert all(hashlib.sha256((pathlib.Path('/root/ascend-dev/fnx/fnx')/n).read_bytes()).hexdigest()==s for n,s in ref.items())
assert hashlib.sha256(pathlib.Path('/root/ascend-dev/p275.sh').read_bytes()).hexdigest()==(p/'p275-before.sha256').read_text()
print('R152_SOURCE_HEALTH_VERIFIED',len(ref))
PY
if ! pgrep -f '(^|/)bash /root/ascend-dev/watchdog.sh' >/dev/null;then
 systemd-run --unit=fnx-watchdog --collect /bin/bash /root/ascend-dev/watchdog.sh
fi
systemctl start model-switch.service
echo CANONICAL_R152_RECOVERED_AFTER_HOST_REBOOT
