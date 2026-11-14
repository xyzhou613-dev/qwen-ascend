"""Container-side W8A8-only launch, using frozen runtime and separate evidence."""
import json,os,subprocess,time
from pathlib import Path
P=Path(__file__).resolve().parent
assert (P/'lease.active').exists() and (P/'attribution-pause.json').exists()
assert time.time()<int((P/'window-start.txt').read_text())+17400
for proc in Path('/proc').glob('[0-9]*'):
    try:argv=proc.joinpath('cmdline').read_bytes().split(b'\0')
    except (FileNotFoundError,PermissionError,ProcessLookupError):continue
    assert b'fnx.server' not in argv and b'fnx.pipeline_worker' not in argv,'REFUSED_SECOND_ENGINE'
original=json.loads((P/'original-private.json').read_text());env=original['env'].copy()
env.update(PYTHONPATH=str(P/'smooth')+':'+str(P)+':'+env.get('PYTHONPATH',''),FNX_PORT='19002',
    R157_DIR=str(P),R157_PHASE='smooth',FNX_DENSE_W8A8='1',FNX_KMAJOR_DOWN='0',FNX_HEAD_A8W8='0',
    FNX_W8A8_CALIB_DIR='/models/fnx-diagnostics/r154-w8a8-gemv-20261113/private-calibration',
    FNX_PIPELINE_LOG=str(P/'attribution-worker.log'))
with (P/'attribution-server.log').open('ab') as log:
    child=subprocess.Popen(original['cmd'],cwd=P/'smooth',env=env,stdin=subprocess.DEVNULL,
        stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
(P/'attribution.pid').write_text(str(child.pid)+'\n')
print('ATTRIBUTION_ONLY_W8A8_STARTED',child.pid,flush=True)
