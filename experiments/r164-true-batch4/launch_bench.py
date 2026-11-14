import os,subprocess,sys
from config import P,environment
assert (P/'lease.active').exists(),'EXCLUSIVE_WINDOW_NOT_ACTIVE'
b=int(sys.argv[1]);env=environment('0,1',b)
env['FNX_CTX']=os.environ.get('R164_CTX','262144')
child=subprocess.Popen([sys.executable,str(P/'bench.py')],env=env,cwd=P,
    stdout=open(P/f'batch{b}.log','a'),stderr=subprocess.STDOUT,start_new_session=True)
(P/'bench-pid.json').write_text(__import__('json').dumps(dict(pid=child.pid,batch=b)))
try:rc=child.wait(timeout=1650)
except subprocess.TimeoutExpired:child.kill();child.wait();rc=124
raise SystemExit(rc)
