"""Stop only task-owned model/test processes, without importing the NPU runtime."""
import json,os,signal,time
from pathlib import Path
P=Path(__file__).resolve().parent
targets=set()
for n in ('original.pid','server.pid','production.pid'):
 try:
  pid=int((P/n).read_text());q=Path('/proc')/str(pid)
  cmd=q.joinpath('cmdline').read_bytes().split(b'\0');cwd=str(q.joinpath('cwd').resolve())
  if b'fnx.server' in cmd and cwd in (str(P),'/root/ascend-dev/fnx'):targets.add(pid)
 except (FileNotFoundError,ValueError,PermissionError,ProcessLookupError):pass
for q in Path('/proc').iterdir():
 if not q.name.isdigit() or int(q.name)==os.getpid():continue
 try:
  cmd=q.joinpath('cmdline').read_bytes().split(b'\0')
  if b'-m' in cmd and b'fnx.server' in cmd and str(q.joinpath('cwd').resolve()) in (str(P),'/root/ascend-dev/fnx'):
   targets.add(int(q.name))
  if any(x.decode(errors='replace') in [str(P/n) for n in ('full_layer_probe.py','probe.py','norm_probe.py')] for x in cmd):targets.add(int(q.name))
 except (FileNotFoundError,PermissionError,ProcessLookupError):pass
workers=set()
for q in Path('/proc').iterdir():
 if not q.name.isdigit():continue
 try:
  ppid=int(next(l.split()[1] for l in q.joinpath('status').read_text().splitlines() if l.startswith('PPid:')))
  if ppid in targets and b'fnx.pipeline_worker' in q.joinpath('cmdline').read_bytes().split(b'\0'):workers.add(int(q.name))
 except (FileNotFoundError,PermissionError,ProcessLookupError,StopIteration):pass
targets|=workers
births={pid:Path('/proc',str(pid),'stat').read_text().split()[21] for pid in targets if Path('/proc',str(pid),'stat').exists()}
for sig in (signal.SIGTERM,signal.SIGKILL):
 for pid in targets:
  try:
   if Path('/proc',str(pid),'stat').read_text().split()[21]==births.get(pid):os.kill(pid,sig)
  except (FileNotFoundError,ProcessLookupError):pass
 if sig==signal.SIGTERM:
  for _ in range(50):
   if all(not Path('/proc',str(pid)).exists() for pid in targets):break
   time.sleep(.2)
print('TASK_OWNED_STOP',sorted(targets),flush=True)
