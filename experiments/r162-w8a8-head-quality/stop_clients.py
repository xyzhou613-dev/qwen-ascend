"""Cancel checkpoint clients before removing this run's code judges."""
import json,os,signal,time
from pathlib import Path
P=Path(__file__).resolve().parent
names={'quality_eval.py','quality_client.py','http_bench.py','reuse_check.py'}
targets={}
for p in Path('/proc').glob('[0-9]*'):
    if int(p.name)==os.getpid():continue
    try:
        argv=p.joinpath('cmdline').read_bytes().split(b'\0')
        if p.joinpath('cwd').resolve()==P and any(Path(os.fsdecode(x)).name in names for x in argv if x):
            targets[int(p.name)]=p.joinpath('stat').read_text().split()[21]
    except (FileNotFoundError,PermissionError,ProcessLookupError):pass
def send(sig):
    for pid,birth in targets.items():
        try:
            if Path('/proc',str(pid),'stat').read_text().split()[21]==birth:os.kill(pid,sig)
        except (FileNotFoundError,ProcessLookupError):pass
send(signal.SIGTERM)
for _ in range(50):
    if all(not Path('/proc',str(pid)).exists() for pid in targets):break
    time.sleep(.1)
send(signal.SIGKILL)
for _ in range(20):
    alive=[]
    for pid,birth in targets.items():
        try:
            stat=Path('/proc',str(pid),'stat').read_text().split()
            if stat[21]==birth and stat[2]!='Z':alive.append(pid)
        except (FileNotFoundError,ProcessLookupError):pass
    if not alive:break
    time.sleep(.1)
assert not alive,('CHECKPOINT_CLIENT_STILL_RUNNING',alive)
(P/'client-cancellation.json').write_text(json.dumps(dict(epoch=time.time(),client_pids=list(targets),
    judges_not_removed_until_clients_terminated=True),indent=2)+'\n')
print('CHECKPOINT_CLIENTS_TERMINATED',list(targets),flush=True)
