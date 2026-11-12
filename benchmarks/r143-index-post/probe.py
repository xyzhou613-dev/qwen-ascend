import json,math,statistics
from pathlib import Path
import torch,torch_npu
from fnx.index_post import run
from fnx.score import score
P=Path(__file__).resolve().parent
torch.npu.set_device(0);torch.manual_seed(143)
def reference(x,p,heads):
 g=x.shape[1]
 z=score(x,p) if heads==4 else x
 valid=(torch.arange(g,device=x.device,dtype=torch.int64)[None,:]+1)*4<=p[:,None]+1
 return (z/math.sqrt(128)).masked_fill(~valid,-1e30)
rows=[]
with torch.no_grad():
 for case in range(36):
  heads=1 if case%2==0 else 4;t=(1,16,128)[(case//2)%3];g=(2048,4096,16384,65536)[(case//6)%4]
  x=torch.randn(t*heads,g,device='npu:0')*(10**((case%7)-3))
  p=torch.arange(t,device=x.device,dtype=torch.int64)*13+((0,2048,60000,262143)[case%4])
  p=p.clamp_max(262143).contiguous()
  a=reference(x,p,heads)
  ai=a.topk(min(512,g),-1).indices
  for mode in (0,1):
   b=run(x,p,heads,mode);torch.npu.synchronize()
   same=torch.equal(a.view(torch.int32),b.view(torch.int32))
   row=dict(kind='accuracy',case=case,tokens=t,groups=g,heads=heads,mode=mode,exact=same,different=int((a.view(torch.int32)!=b.view(torch.int32)).sum()),topk_exact=torch.equal(ai,b.topk(min(512,g),-1).indices))
   rows.append(row)
  del x,p,a,ai,b
 # Probe wide nonnegative FP32 exponents/mantissas independently of four-head reduction.
 g=16384;t=128
 raw=torch.randint(0,0x7f800000,(t,g),dtype=torch.int32,device='npu:0')
 x=raw.view(torch.float32);p=torch.full((t,),262143,dtype=torch.int64,device=x.device)
 a=reference(x,p,1)
 for mode in (0,1):
  b=run(x,p,1,mode);torch.npu.synchronize()
  rows.append(dict(kind='wide_fp32',mode=mode,values=x.numel(),exact=torch.equal(a.view(torch.int32),b.view(torch.int32)),different=int((a.view(torch.int32)!=b.view(torch.int32)).sum())))
 (P/'probe.json').write_text(json.dumps(rows,indent=2))
 selected=[mode for mode in (0,1) if all(r['exact'] for r in rows if r['mode']==mode)]
 print('EXACT_SCALING_MODES',selected,flush=True)
 for mode in (0,1):print('DIFFERENT',mode,sum(r['different'] for r in rows if r['mode']==mode),flush=True)
 assert selected,'no exact scaling mode; do not deploy'
 mode=selected[0];(P/'selection.json').write_text(json.dumps(dict(mode=mode)))
 del raw,x,p,a,b
 for t,g,heads in ((1,4096,1),(1,65536,1),(128,65536,4)):
  x=torch.randn(t*heads,g,device='npu:0');p=torch.arange(t,device=x.device,dtype=torch.int64)+260000
  graphs={}
  for name,fn in [('baseline',lambda:reference(x,p,heads)),('fused',lambda:run(x,p,heads,mode))]:
   fn();torch.npu.synchronize();graph=torch.npu.NPUGraph()
   with torch.npu.graph(graph):
    for _ in range(8):out=fn()
   graphs[name]=graph
  for sweep in range(3):
   for name in ('baseline','fused'):
    times=[]
    for _ in range(7):
     a=torch.npu.Event(enable_timing=True);b=torch.npu.Event(enable_timing=True);a.record()
     for _ in range(4):graphs[name].replay()
     b.record();b.synchronize();times.append(a.elapsed_time(b)/32)
    row=dict(kind='timing',tokens=t,groups=g,heads=heads,mode=mode,sweep=sweep,path=name,ms=statistics.median(times));rows.append(row);print(row,flush=True)
  del graphs,x,p,out
  torch.npu.empty_cache()
 (P/'probe.json').write_text(json.dumps(rows,indent=2))
print('INDEX_POST_PROBE_COMPLETE',flush=True)
