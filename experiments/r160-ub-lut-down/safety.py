"""Independent absolute-deadline fallback; never extends or resumes a run."""
import json,subprocess,time
from pathlib import Path
P=Path(__file__).resolve().parent
deadline=float((P/'window-start.txt').read_text())+2580
while (P/'lease.active').exists():
    if time.time()>=deadline:
        subprocess.run(['systemctl','start','--no-block','fnx-r159-rescue.service'],check=True)
    time.sleep(3)
print('GUARDIAN_EXIT_LEASE_CLEARED',flush=True)
