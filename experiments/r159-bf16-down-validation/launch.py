"""Launch exactly one private diagnostic Engine, using a fresh private snapshot."""
import json, os, re, subprocess, sys
from pathlib import Path

P=Path(__file__).resolve().parent
phase=sys.argv[1]
assert phase in ('baseline','combined') and (P/'lease.active').exists()
for p in Path('/proc').glob('[0-9]*'):
    try: cmd=p.joinpath('cmdline').read_bytes().split(b'\0')
    except (FileNotFoundError,PermissionError,ProcessLookupError): continue
    assert b'fnx.server' not in cmd and b'fnx.pipeline_worker' not in cmd, 'REFUSED_SECOND_ENGINE'
assert not re.findall(r'\|\s+\d+\s+\d+\s+\|\s+(\d+)\s+\|',subprocess.check_output(['npu-smi','info'],text=True)), 'REFUSED_OCCUPIED_NPU'
d=json.loads((P/'original-private.json').read_text())
env=d['env'].copy()
env.update(PYTHONPATH=str(P/phase)+':'+str(P)+':'+env.get('PYTHONPATH',''), FNX_PORT='19002',
    R157_DIR=str(P), R157_PHASE=phase, FNX_DENSE_W8A8='1' if phase=='combined' else '0',
    FNX_KMAJOR_DOWN='1' if phase=='combined' else '0', FNX_KMAJOR_SO=str(P/phase/'fnx/libfnx_w4_kmajor.so'),
    FNX_BF16_DOWN_TILE=(P/'selected-tile.txt').read_text().strip(),
    FNX_W8A8_CALIB_DIR='/models/fnx-diagnostics/r154-w8a8-gemv-20261113/private-calibration',
    FNX_PIPELINE_LOG=str(P/f'{phase}-worker.log'))
with (P/f'{phase}-server.log').open('ab') as log:
    child=subprocess.Popen(d['cmd'], cwd=P/phase, env=env, stdin=subprocess.DEVNULL,
        stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
(P/f'{phase}.pid').write_text(str(child.pid))
print('STARTED', phase, child.pid, flush=True)
