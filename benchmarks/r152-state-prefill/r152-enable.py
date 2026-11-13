"""Enable the validated FP32 prefill state kernel in an r151-compatible launcher."""
import argparse, os
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('launcher',type=Path);p.add_argument('--check',action='store_true');a=p.parse_args()
raw=a.launcher.read_bytes()
default=b'FNX_STATE_UPDATE=${FNX_STATE_UPDATE:-1}\n'
passed=b'FNX_STATE_UPDATE=$FNX_STATE_UPDATE '
if default in raw:
 assert raw.count(default)==1 and raw.count(passed)==1,'incomplete or duplicate configuration'
 print('ALREADY_ENABLED');raise SystemExit(0)
anchor=b'FNX_NORM_POST=${FNX_NORM_POST:-1}\n';assert raw.count(anchor)==1,'unrecognized defaults'
raw=raw.replace(anchor,anchor+default)
anchor=b'FNX_NORM_POST=$FNX_NORM_POST ';assert raw.count(anchor)==1,'unrecognized environment'
raw=raw.replace(anchor,anchor+passed)
if not a.check:
 tmp=a.launcher.with_name(a.launcher.name+'.r152-next');tmp.write_bytes(raw)
 tmp.chmod(a.launcher.stat().st_mode & 0o777);os.replace(tmp,a.launcher)
print('CHECK_OK' if a.check else 'ENABLED')
