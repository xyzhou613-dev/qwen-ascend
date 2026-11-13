"""One bounded component process; no Engine, and refuse any live model."""
import json,os,re,subprocess,sys
from pathlib import Path

P=Path(__file__).resolve().parent
assert (P/'lease.active').exists()
device=int(sys.argv[1]); assert device in (0,1)
script=sys.argv[2]; assert script in ('component_probe.py','layer_probe.py')
for p in Path('/proc').glob('[0-9]*'):
    try: cmd=p.joinpath('cmdline').read_bytes().split(b'\0')
    except (FileNotFoundError,PermissionError,ProcessLookupError): continue
    assert b'fnx.server' not in cmd and b'fnx.pipeline_worker' not in cmd, 'REFUSED_SHARED_TIMING'
assert not re.findall(r'\|\s+\d+\s+\d+\s+\|\s+(\d+)\s+\|',subprocess.check_output(['npu-smi','info'],text=True)), 'REFUSED_OCCUPIED_NPU'
env=os.environ.copy()
env.update(json.loads(Path('/models/fnx-diagnostics/r155-smooth-tpot-ab-20261113/production-selected-config.json').read_text()))
env.update(ASCEND_RT_VISIBLE_DEVICES=str(device), OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
    PYTHONPATH=str(P/'combined')+':'+str(P), FNX_CTX='8192', R157_DIR=str(P), R157_PHASE='combined',
    FNX_DENSE_W8A8='1', FNX_KMAJOR_SO=str(P/'combined/fnx/libfnx_w4_kmajor.so'),
    FNX_BF16_DOWN_TILE=(P/'selected-tile.txt').read_text().strip() if (P/'selected-tile.txt').exists() else '256',
    FNX_W8A8_CALIB_DIR='/models/fnx-diagnostics/r154-w8a8-gemv-20261113/private-calibration')
raise SystemExit(subprocess.call([sys.executable,str(P/script),str(device)],cwd=P,env=env))
