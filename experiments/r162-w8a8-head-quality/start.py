"""Requires new explicit authorization; never reuse the expired r159 hour."""
import datetime,hashlib,json,subprocess,time,uuid
from pathlib import Path
P=Path(__file__).resolve().parent
authorization=json.loads((P/'authorization.json').read_text())
assert authorization['max_seconds']==21600 and authorization['explicit_user_permission']
assert not (P/'window-start.txt').exists(),'REFUSED_DEADLINE_RESET'
frozen=json.loads((P/'harness-manifest.json').read_text())
assert all(hashlib.sha256((P/name).read_bytes()).hexdigest()==sha for name,sha in frozen.items())
refs=json.loads((P/'canonical-grader-summary.json').read_text());assert refs['all_references_passed'] and refs['completed']==1200
subprocess.run(['/usr/bin/python3',str(P/'preflight.py')],check=True)
start=int(time.time());(P/'window-start.txt').write_text(str(start)+'\n')
(P/'run-id').write_text(uuid.uuid4().hex+'\n');(P/'lease.active').touch()
root=Path('/run/systemd/system');service=root/'fnx-r162-rescue.service';timer=root/'fnx-r162-rescue.timer'
assert not service.exists() and not timer.exists()
service.write_text(f'[Unit]\nDescription=Restore r152 after bounded head quality evaluation\n[Service]\nType=oneshot\nExecStart=/bin/bash {P}/rescue.sh\nTimeoutStartSec=1000\nRestart=on-failure\nRestartSec=15\n')
deadline=datetime.datetime.fromtimestamp(start+20100,datetime.timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
timer.write_text(f'[Timer]\nOnCalendar={deadline}\nAccuracySec=1s\nUnit=fnx-r162-rescue.service\n')
subprocess.run(['systemctl','daemon-reload'],check=True);subprocess.run(['systemctl','start','fnx-r162-rescue.timer'],check=True)
subprocess.run(['systemd-run','--unit=fnx-r162-safety','--property=RuntimeMaxSec=21600','/usr/bin/python3',str(P/'safety.py')],check=True)
subprocess.run(['systemd-run','--unit=fnx-r162-run','--property=RuntimeMaxSec=20120','--property=TimeoutStopSec=10',
    '--property=KillMode=control-group','/bin/bash','-c',f'exec /bin/bash "{P}/controller.sh" >> "{P}/controller.log" 2>&1'],check=True)
(P/'window.json').write_text(json.dumps(dict(start_epoch=start,restore_trigger_epoch=start+20100,
    backup_epoch=start+20280,hard_cap_epoch=start+21600,absolute_timer=deadline),indent=2)+'\n')
print('R162_NEW_AUTHORIZED_WINDOW_STARTED',deadline,flush=True)
