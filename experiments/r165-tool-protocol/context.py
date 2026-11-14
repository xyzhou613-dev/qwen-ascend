import subprocess
from pathlib import Path
P=Path(__file__).resolve().parent
def restore_context():
    f=Path('/run/systemd/system/model-switch.service.d/r165-exclusive.conf');f.unlink(missing_ok=True)
    subprocess.run(['systemctl','daemon-reload'],check=True)
    subprocess.run(['systemctl','start','model-switch.service'],check=True)
    found=[]
    for p in Path('/proc').glob('[0-9]*'):
        try:
            a=p.joinpath('cmdline').read_bytes().split(b'\0')
            if a and Path(a[0].decode()).name in ('bash','sh') and b'/root/ascend-dev/watchdog.sh' in a:found.append(p.name)
        except (FileNotFoundError,PermissionError,ProcessLookupError):pass
    if not found:subprocess.run(['systemd-run','--unit=fnx-watchdog','--collect','/bin/bash','/root/ascend-dev/watchdog.sh'],check=True)
    assert len(found)<=1,'DUPLICATE_WATCHDOG'
