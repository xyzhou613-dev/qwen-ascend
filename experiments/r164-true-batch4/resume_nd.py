"""Resume within the original deadline after verified early restoration."""
import hashlib,json,shutil,subprocess,time
from pathlib import Path
P=Path(__file__).resolve().parent
a=json.loads((P/'authorization.json').read_text());assert a['explicit_user_permission'] and a['max_seconds']==3600
w=json.loads((P/'window.json').read_text());remaining=w['restore_trigger_epoch']-time.time()
assert remaining>1000,'TOO_LATE_TO_RESUME'
assert not (P/'lease.active').exists()
assert json.loads((P/'restore-complete.json').read_text())['within_hour']
assert json.loads((P/'probe-nd.json').read_text())['status']=='passed'
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
assert all(sha(P/k)==v for k,v in json.loads((P/'harness-manifest.json').read_text()).items())
for unit in ('fnx-r164-run.service','fnx-r164-rescue.service'):
    assert subprocess.check_output(['systemctl','show',unit,'-p','ActiveState','--value'],text=True).strip() not in ('active','activating','deactivating')
assert subprocess.check_output(['systemctl','is-active','fnx-r164-rescue.timer'],text=True).strip()=='active'
subprocess.run(['python3',str(P/'idle_guard.py')],check=True)
archive=P/'attempt-scatter';archive.mkdir(exist_ok=True)
for n in ('batch4-results.json','batch4-graph-check.json','batch4-failure.json','batch4.log','controller.log','restore.log','restore-complete.json','restored-health.json'):
    f=P/n
    if f.exists() and not (archive/n).exists():shutil.copy2(f,archive/n)
for n in ('batch4-results.json','batch4-graph-check.json','batch4-failure.json','restore-complete.json'):(P/n).unlink(missing_ok=True)
names=('write_nd.py','probe_nd.py','prepare_nd.py','batch_ops_nd.py','bench_nd.py','launch_nd.py','controller_nd.sh','resume_nd.py','probe-nd.json')
(P/'nd-manifest.json').write_text(json.dumps({n:sha(P/n) for n in names},indent=2))
(P/'nd-resume.json').write_text(json.dumps(dict(epoch=time.time(),reason='KV scatter workspace OOM; exact sparse write probe passed',
    original_window=w,remaining_to_restore_s=remaining,original_deadlines_unchanged=True,
    prefill_chunk=2048,short_repetitions=3,long_repetitions=1),indent=2))
(P/'lease.active').touch()
try:
    load=subprocess.check_output(['systemctl','show','fnx-r164-safety.service','-p','LoadState','--value'],text=True).strip()
    if load=='not-found':
        subprocess.run(['systemd-run','--unit=fnx-r164-safety','--property=RuntimeMaxSec='+str(int(w['hard_cap_epoch']-time.time())),
            'python3',str(P/'safety.py')],check=True)
    else:subprocess.run(['systemctl','restart','fnx-r164-safety.service'],check=True)
    subprocess.run(['systemd-run','--unit=fnx-r164-run-nd','--property=RuntimeMaxSec='+str(int(remaining)+10),
        '--property=TimeoutStopSec=10','--property=KillMode=control-group','/bin/bash','-c',
        f'exec /bin/bash "{P}/controller_nd.sh" >> "{P}/controller-nd.log" 2>&1'],check=True)
except Exception:
    subprocess.run(['systemctl','start','--no-block','fnx-r164-rescue.service']);raise
print('RESUMED_WITH_UNCHANGED_ABSOLUTE_DEADLINE',remaining)
