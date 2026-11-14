"""Remove the optional control unit only after canonical service restoration."""
import hashlib,json,subprocess,time,urllib.request
from pathlib import Path
P=Path(__file__).resolve().parent;assert not (P/'lease.active').exists()
h=json.load(urllib.request.urlopen('http://127.0.0.1:9002/health',timeout=5))
assert h['service_revision']=='r152' and h['worker_alive'] and 'diagnostic_variant' not in h
unit='fnx-r162-attribution.service'
state=lambda key:subprocess.check_output(['systemctl','show',unit,'-p',key,'--value'],text=True).strip()
if state('LoadState')!='not-found':subprocess.run(['systemctl','stop',unit],check=True)
assert state('ActiveState') not in ('active','activating','deactivating')
f=Path('/run/systemd/system')/unit
if f.exists():
    assert f.read_bytes()==(P/'attribution-unit.txt').read_bytes();f.unlink()
subprocess.run(['systemctl','daemon-reload'],check=True)
subprocess.run(['systemctl','reset-failed',unit],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
assert not f.exists()
helpers=[]
for proc in Path('/proc').glob('[0-9]*'):
    try:argv=proc.joinpath('cmdline').read_bytes().split(b'\0')
    except (FileNotFoundError,PermissionError,ProcessLookupError):continue
    if any(str(P/n).encode() in argv for n in ('attribution.py','recover_after_attribution.py','launch_attribution.py','start_attribution.py')):helpers.append(int(proc.name))
assert not helpers
frozen=json.loads((P/'attribution-harness-manifest.json').read_text())
assert all(hashlib.sha256((P/n).read_bytes()).hexdigest()==sha for n,sha in frozen.items())
window=json.loads((P/'window.json').read_text());aux=json.loads((P/'attribution-window.json').read_text())
assert aux['original_hard_cap_epoch']==window['hard_cap_epoch'] and aux['original_restore_trigger_epoch']==window['restore_trigger_epoch']
(P/'attribution-final-validation.json').write_text(json.dumps(dict(epoch=time.time(),own_unit_removed=True,
    helper_processes_absent=True,source_harness_unchanged=True,original_deadline_unchanged=True,
    r152_already_restored=True),indent=2)+'\n')
print('ATTRIBUTION_UNIT_AND_HELPERS_CLEANED_ORIGINAL_DEADLINE_PRESERVED')
