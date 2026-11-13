import json,os,subprocess,sys
from pathlib import Path
P=Path(__file__).resolve().parent
env=os.environ.copy();env.update(json.load(open('/models/fnx-diagnostics/r155-smooth-tpot-ab-20261113/production-selected-config.json')))
env.update(ASCEND_RT_VISIBLE_DEVICES=sys.argv[1],OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
 PYTHONPATH=str(P/'candidate')+':'+str(P),FNX_CTX='8192',FNX_KMAJOR_SO=str(P/'libfnx_w4_kmajor.so'))
raise SystemExit(subprocess.call([sys.executable,str(P/sys.argv[2])]+sys.argv[3:],cwd=P,env=env))
