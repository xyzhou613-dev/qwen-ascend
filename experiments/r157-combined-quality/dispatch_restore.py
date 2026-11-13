"""Cancel owned scoring clients before the independent rescue kills code judges.

Keep the rescue lock while the controller exits so its EXIT trap delegates to
the independent service. Original timer/guardian remain fallback protections.
"""
import fcntl
import json
import os
import signal
import subprocess
import time
from pathlib import Path

P=Path(__file__).resolve().parent
ALLOWED={'quality_eval.py','quality_client.py','http_bench.py'}


def state(unit):
    return subprocess.check_output(['systemctl','show',unit,'--value','-p','ActiveState'],text=True).strip()


def identity(pid):
    try:
        parts=(Path('/proc')/str(pid)/'stat').read_text().rsplit(')',1)[1].split()
        return parts[19],parts[0]
    except (FileNotFoundError,ProcessLookupError):
        return None


def live(pid,birth):
    current=identity(pid)
    return current is not None and current[0]==birth and current[1]!='Z'


def main():
    record=dict(started_epoch=time.time(),clients=[],lock_acquired=False)
    lock=(P/'rescue.lock').open('a')
    try:
        try:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            record['reason']='an existing rescue holds the lock'
            return
        record['lock_acquired']=True
        if not (P/'lease.active').exists():
            record['reason']='already restored'
            return
        unit=(P/'current-run-unit').read_text().strip()
        assert unit in ('fnx-r157-run','fnx-r157-run-resume')
        record['unit']=unit
        for proc in Path('/proc').glob('[0-9]*'):
            try:
                args=proc.joinpath('cmdline').read_bytes().split(b'\0')
                if not args or not Path(os.fsdecode(args[0])).name.startswith('python'):
                    continue
                if not any(Path(os.fsdecode(a)).name in ALLOWED for a in args if a):
                    continue
                if os.readlink(proc/'cwd')!=str(P) or unit+'.service' not in (proc/'cgroup').read_text():
                    continue
                ident=identity(int(proc.name))
                if ident and ident[1]!='Z':
                    record['clients'].append(dict(pid=int(proc.name),birth=ident[0]))
            except (FileNotFoundError,PermissionError,ProcessLookupError):
                continue
        for r in record['clients']:
            if live(r['pid'],r['birth']):
                try:os.kill(r['pid'],signal.SIGTERM)
                except ProcessLookupError:pass
        until=time.monotonic()+3
        while any(live(r['pid'],r['birth']) for r in record['clients']) and time.monotonic()<until:
            time.sleep(.05)
        for r in record['clients']:
            if live(r['pid'],r['birth']):
                try:os.kill(r['pid'],signal.SIGKILL)
                except ProcessLookupError:pass
        subprocess.run(['systemctl','--no-block','stop',unit],check=True)
        until=time.monotonic()+10
        while state(unit) in ('active','activating','deactivating') and time.monotonic()<until:
            time.sleep(.1)
        if state(unit) in ('active','activating','deactivating'):
            subprocess.run(['systemctl','kill','--kill-who=all','--signal=SIGKILL',unit],check=True)
            time.sleep(.3)
        record['clients_stopped']=not any(live(r['pid'],r['birth']) for r in record['clients'])
        record['unit_state_after_stop']=state(unit)
        assert record['clients_stopped']
        assert record['unit_state_after_stop'] not in ('active','activating','deactivating')
        record['code_judges_not_killed_until_clients_stopped']=True
    except Exception as e:
        record['error']=repr(e)
        raise
    finally:
        record['finished_epoch']=time.time()
        try:
            (P/'cap-cancellation.json').write_text(json.dumps(record,indent=2)+'\n')
        finally:
            lock.close()
            subprocess.run(['systemctl','--no-block','start','fnx-r157-rescue.service'],check=True)


if __name__=='__main__':
    main()
