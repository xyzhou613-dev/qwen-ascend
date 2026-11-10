"""Enable cumulative decode optimizations in an existing r135-or-newer profile."""
from pathlib import Path
import sys
p=Path(sys.argv[1]) if len(sys.argv)>1 else Path(__file__).resolve().parent/'p275.sh'
b=p.read_bytes()
for name,value in (('FNX_INJECT_NATIVE','1'),('FNX_EXPERT_GATHER','1'),('FNX_DECODE_DN_GMM','0')):
 target=(name+'='+value).encode();other=(name+'='+('0' if value=='1' else '1')).encode()
 if target in b:continue
 if other in b:
  assert b.count(other)==1;b=b.replace(other,target)
 else:
  a=b'FNX_CONV_PREFILL=1';assert b.count(a)==1,'Set the three environment flags explicitly in this profile'
  b=b.replace(a,a+b' '+target)
p.write_bytes(b);print('Enabled r139 decode profile in',p)
