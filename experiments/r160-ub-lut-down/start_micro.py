"""Continue only within the already approved r159 absolute hour; never reset it."""
import datetime,hashlib,json,subprocess,time,urllib.request
from pathlib import Path
P=Path(__file__).resolve().parent
assert not (P/'window.json').exists()
old=json.loads(Path('/models/fnx-diagnostics/r159-bf16-down-validation-20261114/window.json').read_text())
assert time.time()<old['restore_trigger_epoch']-150,'NOT_ENOUGH_TIME_IN_ORIGINAL_AUTHORIZATION'
for d in (0,1):
    r=json.loads((P/f'numeric-device{d}.json').read_text())
    assert r['status']=='numeric_pass' and r['bitwise_equal_to_r158'] and r['experts_covered']==512
for f in P.glob('*.py'):compile(f.read_bytes(),str(f),'exec')
for f in P.glob('*.sh'):subprocess.run(['bash','-n',str(f)],check=True)
h=json.load(urllib.request.urlopen('http://127.0.0.1:9002/health',timeout=5))
assert h['service_revision']=='r152' and h['worker_alive']
assert subprocess.check_output(['systemctl','is-active','model-switch.service'],text=True).strip()=='active'
manifest={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in P.iterdir() if f.suffix in ('.py','.sh','.cpp','.so')}
(P/'source-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
window=dict(old,micro_started_epoch=time.time(),scope='two-card component timing only; no Engine constructed',
    approval_basis='user continue instruction within previously approved <=60min hour',original_deadlines_retained=True)
(P/'window.json').write_text(json.dumps(window,indent=2)+'\n')
(P/'window-start.txt').write_text(str(old['start_epoch'])+'\n')
(P/'lease.active').touch()
root=Path('/run/systemd/system')
service=root/'fnx-r160-rescue.service';timer=root/'fnx-r160-rescue.timer'
assert not service.exists() and not timer.exists()
service.write_text(f'[Unit]\nDescription=Restore canonical r152 after bounded LUT component timing\n[Service]\nType=oneshot\nExecStart=/bin/bash {P}/rescue.sh\nTimeoutStartSec=1000\nRestart=on-failure\nRestartSec=15\n')
deadline=datetime.datetime.fromtimestamp(old['restore_trigger_epoch'],datetime.timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
timer.write_text(f'[Timer]\nOnCalendar={deadline}\nAccuracySec=1s\nUnit=fnx-r160-rescue.service\n')
subprocess.run(['systemctl','daemon-reload'],check=True)
subprocess.run(['systemctl','start','fnx-r160-rescue.timer'],check=True)
subprocess.run(['systemd-run','--unit=fnx-r160-safety','--property=RuntimeMaxSec='+str(int(old['hard_cap_epoch']-time.time())),
    '/usr/bin/python3',str(P/'safety.py')],check=True)
remaining=int(old['restore_trigger_epoch']-time.time()-5)
subprocess.run(['systemd-run','--unit=fnx-r160-run','--property=RuntimeMaxSec='+str(remaining),
    '--property=TimeoutStopSec=10','--property=KillMode=control-group','/bin/bash','-c',
    f'exec /bin/bash "{P}/micro_controller.sh" >> "{P}/controller.log" 2>&1'],check=True)
print('COMPONENT_ONLY_ORIGINAL_DEADLINES',window,flush=True)
