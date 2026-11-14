"""Arm a post-full-quality control within the existing authorization/deadline."""
import hashlib,json,subprocess,time
from pathlib import Path
P=Path(__file__).resolve().parent
assert (P/'lease.active').exists() and not (P/'attribution-pause.json').exists()
start=int((P/'window-start.txt').read_text());now=time.time()
assert now<start+16800
assert json.loads((P/'authorization.json').read_text())['explicit_user_permission']
assert subprocess.check_output(['systemctl','show','fnx-r162-run.service','-p','ActiveState','--value'],text=True).strip()=='active'
names=('attribution.py','launch_attribution.py','recover_after_attribution.py','start_attribution.py')
for name in names:compile((P/name).read_bytes(),str(P/name),'exec')
frozen={n:hashlib.sha256((P/n).read_bytes()).hexdigest() for n in names}
assert not (P/'attribution-harness-manifest.json').exists()
(P/'attribution-harness-manifest.json').write_text(json.dumps(frozen,indent=2)+'\n')
unit=Path('/run/systemd/system/fnx-r162-attribution.service');assert not unit.exists()
seconds=max(1,int(start+17400-now))
contents=(f'[Unit]\nDescription=Optional W8A8 same-slice control after all r162 tasks\n'
    f'[Service]\nType=simple\nWorkingDirectory={P}\n'
    f'ExecStart=/bin/bash -c "exec /usr/bin/python3 {P}/attribution.py >> {P}/attribution.log 2>&1"\n'
    f'ExecStopPost=/usr/bin/python3 {P}/recover_after_attribution.py\n'
    f'RuntimeMaxSec={seconds}\nTimeoutStopSec=10\nKillMode=control-group\n')
unit.write_text(contents);(P/'attribution-unit.txt').write_text(contents)
subprocess.run(['systemctl','daemon-reload'],check=True);subprocess.run(['systemctl','start',unit.name],check=True)
(P/'attribution-window.json').write_text(json.dumps(dict(armed_epoch=now,source_manifest=frozen,
    earliest_action_requires_all_1200_complete=True,pause_cutoff_epoch=start+16800,
    auxiliary_cutoff_epoch=start+17400,original_restore_trigger_epoch=start+20100,
    original_hard_cap_epoch=start+21600,original_deadline_unchanged=True,
    exploratory_only_no_quality_threshold_change=True),indent=2)+'\n')
print('POST_FULL_PAIRING_CONTROL_ARMED_ORIGINAL_WINDOW_UNCHANGED',flush=True)
