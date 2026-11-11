from pathlib import Path
import json,statistics,torch,torch_npu
from fnx import moe_epilogue,weighted_reduce,post_ops
P=Path(__file__).resolve().parent
torch.npu.set_device(0);torch.manual_seed(141141)
y=torch.empty((10,25600),device='npu:0',dtype=torch.bfloat16)
w=torch.empty((1,10),device=y.device,dtype=y.dtype);shared=torch.empty((1,2560),device=y.device,dtype=y.dtype);gate=torch.empty((1,1),device=y.device,dtype=y.dtype)
dg=torch.arange(10,device=y.device,dtype=torch.int64)
def baseline():
 diagonal=y.view(10,10,2560)[dg,dg]
 return post_ops.moe_merge(weighted_reduce.reduce(diagonal,w),shared,gate)
def fused():return moe_epilogue.merge(y,w,shared,gate)
rows=[]
with torch.no_grad():
 for i in range(128):
  y.normal_(std=(.01 if i%2 else 10));shared.normal_(std=.1)
  w.copy_(torch.softmax(torch.randn_like(w).float()*3,-1).bfloat16());gate.normal_(std=8)
  a=baseline();b=fused();torch.npu.synchronize()
  row=dict(kind='accuracy',step=i,exact=torch.equal(a.view(torch.int16),b.view(torch.int16)),max_abs=(a.float()-b.float()).abs().max().item());rows.append(row)
 (P/'epilogue-probe.json').write_text(json.dumps(rows,indent=2))
 assert all(r['exact'] for r in rows),[r for r in rows if not r['exact']][:5]
 print('128_DYNAMIC_EPILOGUE_EXACT',flush=True)
 graphs={}
 for name,fn in [('baseline',baseline),('fused',fused)]:
  fn();torch.npu.synchronize();g=torch.npu.NPUGraph()
  with torch.npu.graph(g):
   for _ in range(64):out=fn()
  graphs[name]=g
 for _ in range(16):
  for g in graphs.values():g.replay()
 torch.npu.synchronize()
 for sweep in range(5):
  for name in ('baseline','fused'):
   times=[]
   for _ in range(9):
    a=torch.npu.Event(enable_timing=True);b=torch.npu.Event(enable_timing=True);a.record()
    for _ in range(4):graphs[name].replay()
    b.record();b.synchronize();times.append(a.elapsed_time(b)/256)
   row=dict(kind='timing',sweep=sweep,mode=name,ms=statistics.median(times));rows.append(row);print(row,flush=True)
 (P/'epilogue-probe.json').write_text(json.dumps(rows,indent=2))
print('EPILOGUE_PROBE_COMPLETE',flush=True)
