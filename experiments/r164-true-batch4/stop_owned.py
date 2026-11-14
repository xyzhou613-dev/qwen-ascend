"""Recorded original PIDs/births, or this unique diagnostic directory only."""
import json,os,signal,time
from pathlib import Path
P=Path(__file__).resolve().parent
records=json.loads((P/'ownership.json').read_text()) if (P/'ownership.json').exists() else []
targets={}
for proc in Path('/proc').iterdir():
    if not proc.name.isdigit() or int(proc.name)==os.getpid():continue
    try:
        cmd=proc.joinpath('cmdline').read_bytes().split(b'\0');cwd=str(proc.joinpath('cwd').resolve());birth=proc.joinpath('stat').read_text().split()[21]
        captured=(P/'original-boot-id').read_text().strip()==Path('/proc/sys/kernel/random/boot_id').read_text().strip() and (P/'stop-requested').exists() and any(int(proc.name)==r['pid'] and birth==r['birth'] and cwd==r['cwd'] for r in records)
        diagnostic=any(str(P/n).encode() in cmd for n in ('bench.py','launch_bench.py'))
        auxiliary=False
        if captured or diagnostic or auxiliary:targets[int(proc.name)]=birth
    except (FileNotFoundError,PermissionError,ProcessLookupError):pass
for sig in (signal.SIGTERM,signal.SIGKILL):
    for pid,birth in targets.items():
        try:
            if Path('/proc',str(pid),'stat').read_text().split()[21]==birth:os.kill(pid,sig)
        except (FileNotFoundError,ProcessLookupError):pass
    if sig==signal.SIGTERM:
        for _ in range(450):
            if all(not Path('/proc',str(pid)).exists() for pid in targets):break
            time.sleep(.2)
print('OWNED_PROCESSES_STOPPED',sorted(targets),flush=True)
