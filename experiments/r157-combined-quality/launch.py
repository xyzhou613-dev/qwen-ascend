import json,os,subprocess,sys
from pathlib import Path
P=Path(__file__).resolve().parent;phase=sys.argv[1];assert phase in ('baseline','smooth','combined','rollback')
active=[]
for proc in Path('/proc').glob('[0-9]*'):
 try:
  cmd=proc.joinpath('cmdline').read_bytes().split(b'\0')
  if b'fnx.server' in cmd or b'fnx.pipeline_worker' in cmd:active.append(proc.name)
 except (FileNotFoundError,ProcessLookupError,PermissionError):pass
assert not active,('REFUSED_SECOND_ENGINE',active)
d=json.load(open(P/'original-private.json'));env=d['env'].copy();cwd=P/('baseline' if phase=='rollback' else phase)
env.update(PYTHONPATH=str(cwd)+':'+str(P)+':'+env.get('PYTHONPATH',''),FNX_PORT='19002',R157_DIR=str(P),R157_PHASE=phase,
 FNX_DENSE_W8A8='0' if phase in ('baseline','rollback') else '1',FNX_KMAJOR_DOWN='1' if phase=='combined' else '0',
 FNX_KMAJOR_SO=str(cwd/'fnx/libfnx_w4_kmajor.so'),FNX_W8A8_CALIB_DIR='/models/fnx-diagnostics/r154-w8a8-gemv-20261113/private-calibration',
 FNX_PIPELINE_LOG=str(P/f'{phase}-worker.log'))
with (P/f'{phase}-server.log').open('ab') as f:
 child=subprocess.Popen(d['cmd'],cwd=cwd,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
(P/f'{phase}.pid').write_text(str(child.pid));print('STARTED',phase,child.pid,flush=True)
