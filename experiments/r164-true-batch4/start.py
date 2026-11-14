"""New explicit 60-minute window required; no authorization file is created."""
import datetime,hashlib,json,subprocess,time
from pathlib import Path
P=Path(__file__).resolve().parent
a=json.loads((P/'authorization.json').read_text())
assert a['explicit_user_permission'] and a['max_seconds']==3600
assert not (P/'window-start.txt').exists(),'NO_DEADLINE_RESET'
manifest=json.loads((P/'harness-manifest.json').read_text())
assert all(hashlib.sha256((P/k).read_bytes()).hexdigest()==v for k,v in manifest.items())
for d in (0,1):assert json.loads((P/f'probe-device{d}.json').read_text())['status']=='passed'
assert json.loads((P/'cpu-preflight.json').read_text())['status']=='passed'
c=json.loads((P/'canonical-manifest.json').read_text())
base=Path('/root/ascend-dev/fnx/fnx')
assert all(hashlib.sha256((base/k).read_bytes()).hexdigest()==v for k,v in c['files'].items())
assert hashlib.sha256(Path('/root/ascend-dev/p275.sh').read_bytes()).hexdigest()==c['p275']
subprocess.run(['python3',str(P/'idle_guard.py')],check=True)
root=Path('/run/systemd/system')
service=root/'fnx-r164-rescue.service';timer=root/'fnx-r164-rescue.timer'
assert not service.exists() and not timer.exists()
start=int(time.time());(P/'window-start.txt').write_text(str(start));(P/'lease.active').touch()
service.write_text(f'[Unit]\nDescription=Restore r152 after true batch4 test\n[Service]\nType=oneshot\nExecStart=/bin/bash {P}/rescue.sh\nTimeoutStartSec=1000\nRestart=on-failure\nRestartSec=15\n')
deadline=datetime.datetime.fromtimestamp(start+2400,datetime.timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
timer.write_text(f'[Timer]\nOnCalendar={deadline}\nAccuracySec=1s\nUnit=fnx-r164-rescue.service\n')
subprocess.run(['systemctl','daemon-reload'],check=True)
subprocess.run(['systemctl','start','fnx-r164-rescue.timer'],check=True)
subprocess.run(['systemd-run','--unit=fnx-r164-safety','--property=RuntimeMaxSec=3600','python3',str(P/'safety.py')],check=True)
subprocess.run(['systemd-run','--unit=fnx-r164-run','--property=RuntimeMaxSec=2420','--property=TimeoutStopSec=10',
    '--property=KillMode=control-group','/bin/bash','-c',f'exec /bin/bash "{P}/controller.sh" >> "{P}/controller.log" 2>&1'],check=True)
(P/'window.json').write_text(json.dumps(dict(start_epoch=start,restore_trigger_epoch=start+2400,
    independent_guard_epoch=start+2460,hard_cap_epoch=start+3600),indent=2))
print('NEW_BATCH4_AUTHORIZED_WINDOW_STARTED',deadline)
