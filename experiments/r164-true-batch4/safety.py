import subprocess,time
from pathlib import Path
P=Path(__file__).resolve().parent
deadline=int((P/'window-start.txt').read_text())+2460
while (P/'lease.active').exists():
    if time.time()>=deadline:subprocess.run(['systemctl','start','--no-block','fnx-r164-rescue.service'],check=True)
    time.sleep(3)
