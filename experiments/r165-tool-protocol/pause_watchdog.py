import os,signal
from pathlib import Path
for p in Path('/proc').glob('[0-9]*'):
    try:
        argv=p.joinpath('cmdline').read_bytes().split(b'\0')
        if argv and Path(os.fsdecode(argv[0])).name in ('bash','sh') and b'/root/ascend-dev/watchdog.sh' in argv:
            os.kill(int(p.name),signal.SIGTERM)
    except (FileNotFoundError,PermissionError,ProcessLookupError):pass
