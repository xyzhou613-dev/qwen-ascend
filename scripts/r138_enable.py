"""Enable the cumulative r137/r138 kernels in an existing r135-or-newer profile."""
from pathlib import Path
import sys
profile=Path(sys.argv[1]) if len(sys.argv)>1 else Path(__file__).resolve().parent/'p275.sh'
data=profile.read_bytes()
for name in ('FNX_INJECT_NATIVE','FNX_EXPERT_GATHER'):
    enabled=(name+'=1').encode();disabled=(name+'=0').encode()
    if enabled in data:continue
    if disabled in data:
        assert data.count(disabled)==1
        data=data.replace(disabled,enabled)
    else:
        anchor=b'FNX_CONV_PREFILL=1'
        assert data.count(anchor)==1,'Unsupported profile; set the two environment flags explicitly'
        data=data.replace(anchor,anchor+b' '+enabled)
profile.write_bytes(data)
print('Enabled r137 residual and r138 expert gather kernels in',profile)
