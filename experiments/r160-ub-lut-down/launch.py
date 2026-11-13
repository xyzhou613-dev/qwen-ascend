import json,os,subprocess,sys
from pathlib import Path
P=Path(__file__).resolve().parent;device=int(sys.argv[1]);assert device in (0,1)
extra=sys.argv[2:];assert extra in ([],['--timing'])
env=os.environ.copy();env.update(json.loads(Path('/models/fnx-diagnostics/r155-smooth-tpot-ab-20261113/production-selected-config.json').read_text()))
env.update(ASCEND_RT_VISIBLE_DEVICES=str(device),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
    PYTHONPATH='/root/ascend-dev/fnx',FNX_W4_SCALE_CACHE='1',FNX_EXPERT_GATHER='1',FNX_DECODE_DN_GMM='0')
raise SystemExit(subprocess.call([sys.executable,str(P/'probe.py'),str(device),*extra],cwd=P,env=env))
