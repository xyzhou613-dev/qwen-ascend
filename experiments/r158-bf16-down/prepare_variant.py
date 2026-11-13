"""Make an isolated down variant; do not modify any r157 runtime or weight."""
import argparse
import difflib
import hashlib
import json
from pathlib import Path

a=argparse.ArgumentParser();a.add_argument('--source',type=Path,required=True);o=a.parse_args()
p=Path(__file__).resolve().parent
original=o.source.read_text()
s=original.split('\n#include "diag_epilogue.cpp"')[0]
assert s.count('Mul(f,f,sf,TILE);PipeBarrier<PIPE_V>();')==1
anchor='Cast(f,h,RoundMode::CAST_NONE,64*TILE);Cast(sf,s,RoundMode::CAST_NONE,TILE);PipeBarrier<PIPE_V>();'
assert s.count(anchor)==1
addition='''
   // Preserve a BF16-rounded dequantized weight before multiplying by x.
   // hb is no longer needed for its original int4-to-half conversion.
   for(int col=0;col<TILE;col+=64)
    Mul(f[col],f[col],sf[col],64,64,{1,1,1,TILE/8,TILE/8,0});
   PipeBarrier<PIPE_V>();
   auto rounded_weight=hb.Get<bfloat16_t>();
   Cast(rounded_weight,f,RoundMode::CAST_RINT,64*TILE);PipeBarrier<PIPE_V>();
   Cast(f,rounded_weight,RoundMode::CAST_NONE,64*TILE);PipeBarrier<PIPE_V>();'''
s=s.replace(anchor,anchor+addition).replace('   Mul(f,f,sf,TILE);PipeBarrier<PIPE_V>();\n','')
assert 'Mul(f,f,sf,TILE)' not in s
for old,new in [('run_down','run_down_bf16w'),('down64','down64_bf16w'),('down128','down128_bf16w'),('down256','down256_bf16w'),('fnx_w4_kmajor','fnx_w4_bf16w')]:
 s=s.replace(old,new)
target=p/'bf16_weight_down.cpp';assert not target.exists();target.write_text(s+'\n')
(p/'variant.patch').write_text(''.join(difflib.unified_diff(original.splitlines(True),(s+'\n').splitlines(True),fromfile='r156/w4_kmajor.cpp',tofile='r158/bf16_weight_down.cpp')))
(p/'source-manifest.json').write_text(json.dumps(dict(original_source=str(o.source),original_sha256=hashlib.sha256(o.source.read_bytes()).hexdigest(),variant_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),scope='standalone numerical hypothesis; not installed or quality-accepted'),indent=2)+'\n')
print('ISOLATED_BF16_WEIGHT_VARIANT_PREPARED')
