"""Independent rollback; never touch an accepted release."""
import fcntl,json,subprocess,time,urllib.request
from pathlib import Path
from context import restore_context
P=Path(__file__).resolve().parent
lock=open(P/'recovery.lock','w')
try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
except BlockingIOError:raise SystemExit(0)
state=open(P/'state.lock','w');fcntl.flock(state,fcntl.LOCK_EX)
if not (P/'lease.active').exists() or (P/'accepted.json').exists():raise SystemExit(0)
(P/'rollback-requested.json').write_text(json.dumps(dict(epoch=time.time())))
fcntl.flock(state,fcntl.LOCK_UN)
subprocess.run(['systemctl','stop','fnx-r165-deploy.service'])
if not (P/'stop-requested').exists():
    restore_context()
    h=json.load(urllib.request.urlopen('http://127.0.0.1:9002/health',timeout=5));assert h['worker_alive']
    (P/'rollback-complete.json').write_text(json.dumps(dict(status='unchanged_no_stop',epoch=time.time(),health=h),indent=2))
    (P/'original-private.json').unlink(missing_ok=True);(P/'lease.active').unlink();raise SystemExit(0)
subprocess.run(['docker','exec','bole-model-Flash','/root/ascend-dev/venv/bin/python',str(P/'owned.py'),'stop'],check=True,timeout=30)
subprocess.run(['python3',str(P/'apply_release.py'),'rollback'],check=True)
with (P/'rollback-start.log').open('ab') as log:
    subprocess.run(['env','WD=0','bash','/root/ascend-dev/p275.sh'],stdout=log,stderr=subprocess.STDOUT,check=True,timeout=450)
h=json.load(urllib.request.urlopen('http://127.0.0.1:9002/health',timeout=5))
assert h['status']=='ok' and h['worker_alive'] and h['service_revision']=='r152' and not h.get('tool_protocol_revision')
restore_context();subprocess.run(['python3',str(P/'formal_smoke.py')],check=True,timeout=120)
import shutil
shutil.copy2(P/'formal-smoke.json',P/'rollback-formal-smoke.json')
start=json.loads((P/'window.json').read_text())['start_epoch']
(P/'rollback-complete.json').write_text(json.dumps(dict(status='rolled_back',epoch=time.time(),elapsed_s=time.time()-start,health=h),indent=2))
(P/'original-private.json').unlink(missing_ok=True);(P/'lease.active').unlink()
print('R165_OLD_PROTOCOL_RESTORED',flush=True)
