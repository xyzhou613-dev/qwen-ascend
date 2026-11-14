import subprocess,time,json
from pathlib import Path
P=Path(__file__).resolve().parent;w=json.loads((P/'window.json').read_text())
while (P/'lease.active').exists():
    if time.time()>=w['backup_trigger_epoch']:subprocess.run(['systemctl','start','--no-block','fnx-r165-rollback.service'],check=True)
    time.sleep(3)
