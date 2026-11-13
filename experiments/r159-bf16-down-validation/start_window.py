"""Call only after this bounded service interruption is authorized."""
import datetime,json,subprocess,time
from pathlib import Path
P=Path(__file__).resolve().parent
assert not (P/'window-start.txt').exists(), 'REFUSED_DEADLINE_RESET'
policy=json.loads((P/'debug-policy.json').read_text())
assert policy['max_window_seconds']==3600 and policy['restore_trigger_seconds']==2400
assert subprocess.check_output(['systemctl','is-active','model-switch.service'],text=True).strip()=='active'
start=int(time.time()); (P/'window-start.txt').write_text(str(start)+'\n')
(P/'lease.active').touch()
unitroot=Path('/run/systemd/system')
service=unitroot/'fnx-r159-rescue.service';timer=unitroot/'fnx-r159-rescue.timer'
assert not service.exists() and not timer.exists()
service.write_text(f'''[Unit]
Description=Restore canonical r152 after bounded BF16 down validation
[Service]
Type=oneshot
ExecStart=/bin/bash {P}/rescue.sh
TimeoutStartSec=1000
Restart=on-failure
RestartSec=15
''')
deadline=datetime.datetime.fromtimestamp(start+2400,datetime.timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
timer.write_text(f'''[Unit]
Description=Absolute bounded BF16 down recovery deadline
[Timer]
OnCalendar={deadline}
AccuracySec=1s
Unit=fnx-r159-rescue.service
''')
subprocess.run(['systemctl','daemon-reload'],check=True)
subprocess.run(['systemctl','start','fnx-r159-rescue.timer'],check=True)
subprocess.run(['systemd-run','--unit=fnx-r159-safety','--property=RuntimeMaxSec=3600','/usr/bin/python3',str(P/'safety.py')],check=True)
subprocess.run(['systemd-run','--unit=fnx-r159-run','--property=RuntimeMaxSec=2460',
    '--property=TimeoutStopSec=10','--property=KillMode=control-group',
    '--property=StandardOutput=append:'+str(P/'controller.log'),
    '--property=StandardError=append:'+str(P/'controller.log'),'/bin/bash',str(P/'controller.sh')],check=True)
record=dict(start_epoch=start,restore_trigger_epoch=start+2400,backup_trigger_epoch=start+2580,
    hard_cap_epoch=start+3600,absolute_timer=deadline)
(P/'window.json').write_text(json.dumps(record,indent=2)+'\n')
print('BOUNDED_WINDOW_STARTED',record)
