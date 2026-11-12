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
