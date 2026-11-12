"""Add r144 opt-in defaults to the existing 206 launcher; fail closed on drift."""
import argparse
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__);p.add_argument('launcher',type=Path);p.add_argument('--check',action='store_true');a=p.parse_args()
s=a.launcher.read_text()
if 'FNX_NORM_POST=${FNX_NORM_POST:-1}' in s:
 assert 'FNX_NORM_POST=$FNX_NORM_POST' in s and 'FNX_DELTA_REUSE=$FNX_DELTA_REUSE' in s
 print('ALREADY_ENABLED');raise SystemExit(0)
anchor='FNX_INDEX_POST=${FNX_INDEX_POST:-1}'
assert s.count(anchor)==1,'unrecognized launcher defaults'
s=s.replace(anchor,anchor+'\nFNX_NORM_POST=${FNX_NORM_POST:-1}\nFNX_DELTA_REUSE=${FNX_DELTA_REUSE:-1}')
anchor='FNX_CONV_PREFILL=$FNX_CONV_PREFILL FNX_INDEX_POST=$FNX_INDEX_POST'
assert s.count(anchor)==1,'unrecognized launcher environment'
s=s.replace(anchor,'FNX_NORM_POST=$FNX_NORM_POST FNX_DELTA_REUSE=$FNX_DELTA_REUSE '+anchor)
if not a.check:a.launcher.write_text(s)
print('CHECK_OK' if a.check else 'ENABLED')
