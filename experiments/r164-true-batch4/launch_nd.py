"""Same authorized window; register exact child identities for recovery."""
import json,os,subprocess,sys,time
from pathlib import Path
from config import P,environment
assert (P/'lease.active').exists()
w=json.loads((P/'window.json').read_text());remaining=w['restore_trigger_epoch']-time.time()
assert remaining>120
b=int(sys.argv[1]);assert b in (1,4)
env=environment('0,1',b);env['FNX_CHUNK']='2048'
env['FNX_CTX']='262144' if remaining>900 else '16384'
def register(pid):
    proc=Path('/proc',str(pid));records=json.loads((P/'ownership.json').read_text())
    records.append(dict(pid=pid,cwd=str(proc.joinpath('cwd').resolve()),birth=proc.joinpath('stat').read_text().split()[21]))
    (P/'ownership.json').write_text(json.dumps(records,indent=2))
register(os.getpid())
child=subprocess.Popen([sys.executable,str(P/'bench_nd.py')],env=env,cwd=P,
    stdout=open(P/f'batch{b}-nd.log','a'),stderr=subprocess.STDOUT,start_new_session=True)
register(child.pid)
(P/f'batch{b}-nd-launch.json').write_text(json.dumps(dict(batch=b,chunk=2048,ctx=int(env['FNX_CTX']),
    original_window=w,timeout_s=min(1200,int(remaining)-45),pid=child.pid),indent=2))
try:rc=child.wait(timeout=min(1200,int(remaining)-45))
except subprocess.TimeoutExpired:child.kill();child.wait();rc=124
raise SystemExit(rc)
