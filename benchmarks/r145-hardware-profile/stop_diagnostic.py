"""Stop only recorded original and this directory's diagnostic processes."""
import os,signal,time
from pathlib import Path
P=Path(__file__).resolve().parent;targets=set();original=int((P/'original.pid').read_text()) if (P/'original.pid').exists() else -1
for p in Path('/proc').iterdir():
 if not p.name.isdigit() or int(p.name)==os.getpid():continue
 try:
  cmd=p.joinpath('cmdline').read_bytes().split(b'\0');cwd=str(p.joinpath('cwd').resolve())
  if int(p.name)==original and b'fnx.server' in cmd and cwd=='/root/ascend-dev/fnx':targets.add(int(p.name))
  if any(v.decode(errors='replace')==str(P/'profile_model.py') for v in cmd):targets.add(int(p.name))
  if b'fnx.pipeline_worker' in cmd and cwd==str(P):targets.add(int(p.name))
 except (FileNotFoundError,PermissionError,ProcessLookupError):pass
birth={pid:Path('/proc',str(pid),'stat').read_text().split()[21] for pid in targets if Path('/proc',str(pid),'stat').exists()}
for sig in (signal.SIGTERM,signal.SIGKILL):
 for pid in targets:
  try:
   if Path('/proc',str(pid),'stat').read_text().split()[21]==birth.get(pid):os.kill(pid,sig)
  except (FileNotFoundError,ProcessLookupError):pass
 if sig==signal.SIGTERM:
  for _ in range(50):
   if all(not Path('/proc',str(pid)).exists() for pid in targets):break
   time.sleep(.2)
print('OWNED_PROCESSES_STOPPED',sorted(targets),flush=True)
