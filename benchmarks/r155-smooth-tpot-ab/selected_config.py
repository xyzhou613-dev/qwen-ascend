"""Export only documented inference settings, never the complete process ENV."""
import json,re
from pathlib import Path
P=Path(__file__).resolve().parent
pids=[]
for proc in Path('/proc').iterdir():
 if not proc.name.isdigit():continue
 try:
  cmd=proc.joinpath('cmdline').read_bytes().split(b'\0')
  if b'fnx.server' in cmd:pids.append(proc)
 except (FileNotFoundError,PermissionError,ProcessLookupError):pass
assert len(pids)==1
env=dict(s.decode().split('=',1) for s in pids[0].joinpath('environ').read_bytes().split(b'\0') if b'=' in s)
allowed={'OMP_NUM_THREADS','MKL_NUM_THREADS','TASK_QUEUE_ENABLE','ASCEND_RT_VISIBLE_DEVICES','PYTORCH_NPU_ALLOC_CONF'}
allowed|={k for k in env if k.startswith('FNX_') and not re.search(r'API|SECRET|PASSWORD|CREDENTIAL|AUTH|_KEY',k)}
r={k:env[k] for k in sorted(allowed) if k in env}
(P/'production-selected-config.json').write_text(json.dumps(r,indent=2));print('SELECTED_CONFIG_KEYS',len(r),flush=True)
