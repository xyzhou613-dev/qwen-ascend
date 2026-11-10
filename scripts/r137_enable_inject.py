"""Enable the opt-in kernel in an existing r135 profile without changing ports/devices."""
from pathlib import Path
import sys

profile = Path(sys.argv[1]) if len(sys.argv)>1 else Path(__file__).resolve().parent/'p275.sh'
data = profile.read_bytes()
if b'FNX_INJECT_NATIVE=1' in data:
    print('FNX_INJECT_NATIVE already enabled')
    raise SystemExit(0)
if b'FNX_INJECT_NATIVE=0' in data:
    assert data.count(b'FNX_INJECT_NATIVE=0') == 1
    data = data.replace(b'FNX_INJECT_NATIVE=0', b'FNX_INJECT_NATIVE=1')
else:
    anchor = b'FNX_CONV_PREFILL=1'
    assert data.count(anchor) == 1, 'Unsupported profile: add FNX_INJECT_NATIVE=1 to its inference environment manually'
    data = data.replace(anchor, anchor+b' FNX_INJECT_NATIVE=1')
profile.write_bytes(data)
print('Enabled residual injection in', profile)
