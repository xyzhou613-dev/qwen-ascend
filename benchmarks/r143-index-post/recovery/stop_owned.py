import sys,json,os,signal
from pathlib import Path
P=Path(__file__).resolve().parent
mode=sys.argv[1];f=P/('server.pid' if mode=='candidate' else 'production.pid' if mode=='production' else 'original.pid')
if not f.exists():raise SystemExit(0)
pid=int(f.read_text());proc=Path(f'/proc/{pid}')
if not proc.exists():raise SystemExit(0)
cmd=proc.joinpath('cmdline').read_bytes().split(b'\0')
assert b'fnx.server' in cmd,(pid,cmd)
cwd=proc.joinpath('cwd').resolve()
if mode=='candidate':assert cwd==P,cwd
else:assert str(cwd)==json.loads((P/'original-private.json').read_text())['cwd'],cwd
os.kill(pid,signal.SIGTERM);print('STOPPED_OWNED',mode,pid)
