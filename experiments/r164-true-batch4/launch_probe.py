import fcntl,subprocess,sys
from config import P,environment
d=int(sys.argv[1]);assert d in (0,1)
lock=open(P/'probe.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
p=subprocess.Popen([sys.executable,str(P/'probe.py'),str(d)],env=environment(d),cwd=P)
try:rc=p.wait(timeout=400)
except subprocess.TimeoutExpired:p.kill();p.wait();rc=124
raise SystemExit(rc)
