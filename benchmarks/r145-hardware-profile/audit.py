import bisect,collections,json
from pathlib import Path
import analyze as A
P=Path(__file__).resolve().parent
records=json.load(open(P/'runs.json'));out=[]
for rec in records:
 if not rec['metric']:continue
 a,b,n=A.bounds(rec)
 for stage in (0,1):
  k=A.kernels(rec['tag'],stage);counts=[];patterns=[]
  for x,y in zip(rec['rows'],rec['rows'][1:]):
   sub=[r for r in k if r['_start']>=x['end_ns'] and r['_end']<=y['end_ns']]
   counts.append(len(sub));patterns.append(collections.Counter((r['Name'],r['Input Shapes'],r['Accelerator Core']) for r in sub))
  same=all(p==patterns[0] for p in patterns)
  row=dict(tag=rec['tag'],stage=stage,periods=n,kernel_count_min=min(counts),kernel_count_max=max(counts),identical_kernel_signatures=same)
  out.append(row);assert same,row
s=json.load(open(P/'summary.json'))
for context,h in s['hardware'].items():
 for dev,d in h['devices'].items():
  for c in d['channels']:
   if c['kind']=='HBM':assert c['window_coverage']>.99,(context,dev,c)
for r in s['timing'].values():assert r['cross_device_overlap_ms']<.001
(P/'coverage-audit.json').write_text(json.dumps(out,indent=2));print('COUNTER_COVERAGE_AND_CLOCK_ORDER_AUDIT_PASS',len(out))
