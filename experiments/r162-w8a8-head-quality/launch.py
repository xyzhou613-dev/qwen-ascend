import json,os,subprocess,sys
from pathlib import Path
P=Path(__file__).resolve().parent;phase=sys.argv[1];assert phase in ('rollback','smooth','combined')
assert (P/'lease.active').exists()
for p in Path('/proc').glob('[0-9]*'):
    try:c=p.joinpath('cmdline').read_bytes().split(b'\0')
    except (FileNotFoundError,PermissionError,ProcessLookupError):continue
    assert b'fnx.server' not in c and b'fnx.pipeline_worker' not in c,'REFUSED_SECOND_ENGINE'
d=json.loads((P/'original-private.json').read_text());env=d['env'].copy();cwd=P/('baseline' if phase=='rollback' else phase)
env.update(PYTHONPATH=str(cwd)+':'+str(P)+':'+env.get('PYTHONPATH',''),FNX_PORT='19002',R157_DIR=str(P),R157_PHASE=phase,
    FNX_DENSE_W8A8='0' if phase=='rollback' else '1',FNX_KMAJOR_DOWN='0',
    FNX_HEAD_A8W8='1' if phase=='combined' else '0',
    FNX_W8A8_CALIB_DIR='/models/fnx-diagnostics/r154-w8a8-gemv-20261113/private-calibration',
    FNX_PIPELINE_LOG=str(P/f'{phase}-worker.log'))
with (P/f'{phase}-server.log').open('ab') as log:
    child=subprocess.Popen(d['cmd'],cwd=cwd,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
(P/f'{phase}.pid').write_text(str(child.pid));print('STARTED',phase,child.pid,flush=True)
