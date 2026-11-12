from pathlib import Path
P=Path(__file__).resolve().parent;found=[]
for q in Path('/proc').iterdir():
 if not q.name.isdigit():continue
 try:
  cmd=q.joinpath('cmdline').read_bytes().split(b'\0')
  if b'-m' in cmd and b'fnx.server' in cmd and str(q.joinpath('cwd').resolve())=='/root/ascend-dev/fnx':found.append(q.name)
 except (FileNotFoundError,PermissionError,ProcessLookupError):pass
assert len(found)==1,found
(P/'production.pid').write_text(found[0]);print('CANONICAL_PRODUCTION_PID',found[0],flush=True)
