"""Stop this release's canonical FNX only while its deployment lease exists."""
import hashlib,json,os,signal,time
from pathlib import Path
P=Path(__file__).resolve().parent;assert (P/'deployment.active').exists()
m=json.load(open(P/'deployment-backup/manifest.json'));row=next(r for r in m['entries'] if r['name']=='server.py')
assert hashlib.sha256(Path(row['target']).read_bytes()).hexdigest() in (row['old_sha'],row['new_sha'])
targets={}
for proc in Path('/proc').iterdir():
    if not proc.name.isdigit():continue
    try:
        cmd=proc.joinpath('cmdline').read_bytes().split(b'\0');cwd=str(proc.joinpath('cwd').resolve())
        if cwd=='/root/ascend-dev/fnx' and (b'fnx.server' in cmd or b'fnx.pipeline_worker' in cmd):targets[int(proc.name)]=proc.joinpath('stat').read_text().split()[21]
    except (FileNotFoundError,PermissionError,ProcessLookupError):pass
for sig in (signal.SIGTERM,signal.SIGKILL):
    for pid,birth in targets.items():
        try:
            if Path('/proc',str(pid),'stat').read_text().split()[21]==birth:os.kill(pid,sig)
        except (FileNotFoundError,ProcessLookupError):pass
    if sig==signal.SIGTERM:
        for _ in range(50):
            if all(not Path('/proc',str(pid)).exists() for pid in targets):break
            time.sleep(.2)
print('CANONICAL_OWNED_FNXS_STOPPED',sorted(targets),flush=True)
