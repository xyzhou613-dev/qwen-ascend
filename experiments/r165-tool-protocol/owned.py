"""Record/stop only this container's canonical FNX server and its worker."""
import json,os,signal,sys,time
from pathlib import Path
P=Path(__file__).resolve().parent;cwd_expected='/root/ascend-dev/fnx'
def scan():
    rows=[]
    for p in Path('/proc').glob('[0-9]*'):
        try:
            args=p.joinpath('cmdline').read_bytes().split(b'\0')
            if b'fnx.server' not in args and b'fnx.pipeline_worker' not in args:continue
            stat=p.joinpath('stat').read_text().split()
            if stat[2]=='Z':continue
            cwd=str(p.joinpath('cwd').resolve(strict=True))
            assert cwd==cwd_expected,'UNRELATED_FNX_PROCESS'
            env=dict(x.decode().split('=',1) for x in p.joinpath('environ').read_bytes().split(b'\0') if x)
            assert env.get('FNX_PORT')=='9002' and env.get('FNX_CTX')=='262144' and env.get('FNX_NLAYERS')=='48'
            rows.append(dict(pid=int(p.name),birth=stat[21],cwd=cwd,role='server' if b'fnx.server' in args else 'worker'))
        except (FileNotFoundError,PermissionError,ProcessLookupError):pass
    assert len(rows)<=2,'MULTIPLE_FNX_ENGINES'
    return rows
def alive(r):
    try:
        p=Path('/proc',str(r['pid']));stat=p.joinpath('stat').read_text().split()
        return stat[2]!='Z' and stat[21]==r['birth'] and str(p.joinpath('cwd').resolve(strict=True))==r['cwd']
    except (FileNotFoundError,ProcessLookupError):return False
mode=sys.argv[1]
if mode=='check':
    rows=scan();assert len(rows)==2
    print('CANONICAL_OWNERSHIP_PREFLIGHT_PASS',[(r['pid'],r['role']) for r in rows],flush=True);raise SystemExit(0)
assert (P/'lease.active').exists()
if mode=='snapshot':
    rows=scan();assert len(rows)==2 and {r['role'] for r in rows}=={'server','worker'}
    p=Path('/proc',str(next(r['pid'] for r in rows if r['role']=='server')))
    raw=dict(env=[x.decode() for x in p.joinpath('environ').read_bytes().split(b'\0') if x],
             argv=[x.decode() for x in p.joinpath('cmdline').read_bytes().split(b'\0') if x],cwd=cwd_expected)
    f=P/'original-private.json';f.write_text(json.dumps(raw));f.chmod(0o600)
    (P/'ownership.json').write_text(json.dumps(rows,indent=2));(P/'original-boot-id').write_bytes(Path('/proc/sys/kernel/random/boot_id').read_bytes())
elif mode in ('register','stop'):
    assert (P/'original-boot-id').read_bytes()==Path('/proc/sys/kernel/random/boot_id').read_bytes()
    rows=json.loads((P/'ownership.json').read_text());known={(r['pid'],r['birth']) for r in rows}
    for r in scan():
        if (r['pid'],r['birth']) not in known:rows.append(r)
    (P/'ownership.json').write_text(json.dumps(rows,indent=2))
    if mode=='stop':
        for sig,seconds in ((signal.SIGTERM,10),(signal.SIGKILL,5)):
            for r in rows:
                if alive(r):
                    try:os.kill(r['pid'],sig)
                    except ProcessLookupError:pass
            end=time.monotonic()+seconds
            while any(alive(r) for r in rows) and time.monotonic()<end:time.sleep(.2)
        assert not any(alive(r) for r in rows),'OWNED_SERVICE_DID_NOT_EXIT'
        print('OWNED_CANONICAL_SERVICE_STOPPED',flush=True)
else:raise ValueError(mode)
