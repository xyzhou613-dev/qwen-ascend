import json,os
from pathlib import Path
P=Path(__file__).resolve().parent
pids=[]
for p in Path('/proc').iterdir():
    if not p.name.isdigit():continue
    try:
        cmd=p.joinpath('cmdline').read_bytes().split(b'\0')
        if b'-m' in cmd and b'fnx.server' in cmd:pids.append(p)
    except (FileNotFoundError,PermissionError,ProcessLookupError):pass
assert len(pids)==1,[x.name for x in pids]
p=pids[0]
env=dict(x.decode().split('=',1) for x in p.joinpath('environ').read_bytes().split(b'\0') if x)
d=dict(pid=int(p.name),cmd=[x.decode() for x in p.joinpath('cmdline').read_bytes().split(b'\0') if x],cwd=str(p.joinpath('cwd').resolve()),env=env)
f=P/'original-private.json';f.write_text(json.dumps(d));f.chmod(0o600)
(P/'original.pid').write_text(p.name)
print('SNAPSHOT_PID',p.name,'cwd',d['cwd'])

owned=[]
for proc in Path('/proc').iterdir():
 if not proc.name.isdigit():continue
 try:
  cmd=proc.joinpath('cmdline').read_bytes().split(b'\0');cwd=str(proc.joinpath('cwd').resolve())
  parent=int(next(x.split()[1] for x in proc.joinpath('status').read_text().splitlines() if x.startswith('PPid:')))
  if proc.name==p.name or (b'fnx.pipeline_worker' in cmd and parent==int(p.name) and cwd==d['cwd']):
   owned.append(dict(pid=int(proc.name),cwd=cwd,birth=proc.joinpath('stat').read_text().split()[21]))
 except (FileNotFoundError,PermissionError,ProcessLookupError):pass
assert len(owned)==2,owned
(P/'ownership.json').write_text(json.dumps(owned,indent=2))
print('OWNED_SERVER_WORKER_CAPTURED',len(owned))
