"""New bounded repair rollout; never reuse an old performance-test deadline."""
import datetime,hashlib,json,subprocess,time
from pathlib import Path
P=Path(__file__).resolve().parent
assert not (P/'window.json').exists(),'REFUSED_DEADLINE_RESET'
manifest=json.loads((P/'harness-manifest.json').read_text());sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
assert all(sha(P/n)==h for n,h in manifest.items())
m=json.loads((P/'release-manifest.json').read_text());src=Path('/root/ascend-dev/fnx/fnx')
assert all(sha(src/n)==h for n,h in m['old_files'].items()) and not (src/'tool_protocol.py').exists()
assert sha(Path('/root/ascend-dev/p275.sh'))==m['p275_sha256']
assert json.loads((P/'rollback-dryrun.json').read_text())['status']=='passed'
assert subprocess.check_output(['systemctl','is-active','model-switch.service'],text=True).strip()=='active'
assert subprocess.check_output(['systemctl','is-enabled','fnx-boot-restore.service'],text=True).strip()=='enabled'
subprocess.run(['python3',str(P/'idle_guard.py')],check=True)
root=Path('/run/systemd/system');service=root/'fnx-r165-rollback.service';timer=root/'fnx-r165-rollback.timer'
assert not service.exists() and not timer.exists()
start=int(time.time());deadline=datetime.datetime.fromtimestamp(start+600,datetime.timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
(P/'window.json').write_text(json.dumps(dict(start_epoch=start,rollback_trigger_epoch=start+600,backup_trigger_epoch=start+615,
    boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
    operator_cap_epoch=start+1200,cap_seconds=1200,user_instruction='请帮我修复他',scope='protocol repair, necessary idle restart, live validation, rollback on failure'),ensure_ascii=False,indent=2))
service.write_text(f'[Unit]\nDescription=Rollback unaccepted FNX tool protocol repair\n[Service]\nType=oneshot\nExecStart=/usr/bin/python3 {P}/recover.py\nTimeoutStartSec=580\nRestart=on-failure\nRestartSec=5\n')
timer.write_text(f'[Timer]\nOnCalendar={deadline}\nAccuracySec=1s\nUnit=fnx-r165-rollback.service\n')
bootdir=Path('/etc/systemd/system/fnx-boot-restore.service.d');bootdir.mkdir(exist_ok=True)
bootguard=bootdir/'r165-pending-protocol.conf';assert not bootguard.exists()
bootguard.write_text(f'[Service]\nExecStartPre=/usr/bin/python3 {P}/recover_before_boot.py\n')
subprocess.run(['systemd-analyze','verify',str(service),str(timer)],check=True)
subprocess.run(['systemctl','daemon-reload'],check=True)
(P/'lease.active').touch()
try:
    subprocess.run(['systemctl','start','fnx-r165-rollback.timer'],check=True)
    subprocess.run(['systemd-run','--unit=fnx-r165-guard','--collect','--property=RuntimeMaxSec=1200','python3',str(P/'guard.py')],check=True)
    subprocess.run(['systemd-run','--unit=fnx-r165-deploy','--property=RuntimeMaxSec=610','--property=TimeoutStopSec=5',
        '--property=KillMode=control-group','/bin/bash','-c',f'exec /usr/bin/python3 "{P}/deploy.py" >> "{P}/deployment.log" 2>&1'],check=True)
except Exception:
    subprocess.run(['systemctl','start','--no-block','fnx-r165-rollback.service']);raise
print('BOUNDED_PROTOCOL_REPAIR_STARTED',deadline,flush=True)
