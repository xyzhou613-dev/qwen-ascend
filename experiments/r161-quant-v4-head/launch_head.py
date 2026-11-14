"""Bounded paired real layers alongside an idle service; no Engine constructor."""
import json,os,subprocess,sys
from pathlib import Path
P=Path(__file__).resolve().parent;device=int(sys.argv[1]);assert device in (0,1)
env=os.environ.copy();env.update(json.loads(Path('/models/fnx-diagnostics/r155-smooth-tpot-ab-20261113/production-selected-config.json').read_text()))
env.update(ASCEND_RT_VISIBLE_DEVICES=str(device),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',FNX_CTX='8192',
    PYTHONPATH=str(P/'runtime')+':'+str(P),FNX_MOE_GROUP='0',FNX_KMAJOR_DOWN='0',
    FNX_VERIFY_PREFIX_JOURNAL='1',FNX_W8A8_CALIB_DIR='/models/fnx-diagnostics/r154-w8a8-gemv-20261113/private-calibration')
raise SystemExit(subprocess.call([sys.executable,str(P/'head_probe.py'),str(device)],cwd=P,env=env))
