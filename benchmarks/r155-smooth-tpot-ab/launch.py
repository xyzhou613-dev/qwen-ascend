import json,os,subprocess,sys
from pathlib import Path
P=Path(__file__).resolve().parent;phase=sys.argv[1];assert phase in ('baseline','candidate','rollback')
active=[]
for p in Path('/proc').glob('[0-9]*'):
 try:
  cmd=p.joinpath('cmdline').read_bytes().split(b'\0')
  if b'fnx.server' in cmd or b'fnx.pipeline_worker' in cmd:active.append(p.name)
 except (FileNotFoundError,PermissionError,ProcessLookupError):pass
assert not active,('REFUSED_SECOND_ENGINE',active)
d=json.load(open(P/'original-private.json'));env=d['env'].copy();cwd=P/('baseline' if phase=='rollback' else phase)
env.update(PYTHONPATH=str(cwd)+os.pathsep+str(P)+os.pathsep+env.get('PYTHONPATH',''),
    FNX_PORT='19002',FNX_DENSE_W8A8='1' if phase=='candidate' else '0',
    FNX_W8A8_CALIB_DIR='/models/fnx-diagnostics/r154-w8a8-gemv-20261113/private-calibration',
    R155_VARIANT=phase,R155_DIR=str(P),FNX_PIPELINE_LOG=str(P/f'{phase}-worker.log'))
with (P/f'{phase}-server.log').open('ab') as f:
 child=subprocess.Popen(d['cmd'],cwd=cwd,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
(P/f'{phase}.pid').write_text(str(child.pid));print('STARTED_DIAGNOSTIC',phase,child.pid,flush=True)
