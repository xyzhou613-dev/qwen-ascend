"""Resume only this paused controller, then use the original independent rescue."""
import json,os,signal,subprocess
from pathlib import Path
P=Path(__file__).resolve().parent;marker=P/'attribution-pause.json'
if marker.exists():
    row=json.loads(marker.read_text());proc=Path('/proc',str(row['pid']))
    try:
        argv=proc.joinpath('cmdline').read_bytes().split(b'\0')
        if proc.joinpath('stat').read_text().split()[21]==row['birth'] and str(P/'controller.sh').encode() in argv:
            os.kill(row['pid'],signal.SIGCONT)
    except (FileNotFoundError,ProcessLookupError):pass
    if (P/'lease.active').exists():
        subprocess.run(['systemctl','start','--no-block','fnx-r162-rescue.service'],check=True)
print('ATTRIBUTION_RECOVERY_SCOPED_TO_ORIGINAL_WINDOW',flush=True)
