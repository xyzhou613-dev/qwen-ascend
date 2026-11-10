import ctypes,json,statistics,os
from pathlib import Path
import torch,torch_npu
P=Path(__file__).resolve().parent;lib=ctypes.CDLL(os.environ.get('FNX_INJECT_SO',str(P.parents[1]/'fnx/libfnx_inject.so')))
lib.fnx_inject.argtypes=[ctypes.c_void_p]*4+[ctypes.c_int,ctypes.c_void_p]
def native(h,y,g):
 out=torch.empty_like(h);rc=lib.fnx_inject(h.data_ptr(),y.data_ptr(),g.data_ptr(),out.data_ptr(),h.shape[0],torch.npu.current_stream().npu_stream);assert rc==0;return out
def reference(h,y,g):return h+(y.view(-1,1,1,2560)*g.unsqueeze(-1)).reshape(-1,1,10240)
def bench(graph):
 samples=[]
 for _ in range(7):
  a=torch.npu.Event(enable_timing=True);b=torch.npu.Event(enable_timing=True);a.record()
  for _ in range(8):graph.replay()
  b.record();b.synchronize();samples.append(a.elapsed_time(b)/8/128)
 return statistics.median(samples)
@torch.no_grad()
def main():
 torch.npu.set_device(0);torch.manual_seed(137137);rows=[]
 for batch in (1,8,128):
  h=torch.randn(batch,1,10240,device='npu:0',dtype=torch.bfloat16);y=torch.randn(batch,1,2560,device=h.device,dtype=h.dtype);g=torch.rand(batch,1,4,device=h.device,dtype=h.dtype)*2
  graphs={};outputs={}
  for name,fn in (('baseline',reference),('native',native)):
   fn(h,y,g);torch.npu.synchronize();graph=torch.npu.NPUGraph()
   with torch.npu.graph(graph):
    for _ in range(128):outputs[name]=fn(h,y,g)
   graphs[name]=graph
  exact=True;max_abs=0.
  for step in range(32):
   scale=10.**((step%9)-4);h.copy_(torch.randn_like(h)*scale);y.copy_(torch.randn_like(y)*scale);g.copy_(torch.rand_like(g)*2)
   if step==31:h.zero_();y.zero_();g.zero_()
   for graph in graphs.values():graph.replay()
   torch.npu.synchronize();exact &= torch.equal(outputs['baseline'].view(torch.int16),outputs['native'].view(torch.int16));max_abs=max(max_abs,(outputs['baseline'].float()-outputs['native'].float()).abs().max().item())
  timings=[dict(mode=name,ms=bench(graphs[name])) for name in ('baseline','native','baseline','native')]
  row=dict(batch=batch,exact=exact,max_abs=max_abs,dynamic_steps=32,timings=timings);rows.append(row);(P/'inject-probe.json').write_text(json.dumps(rows,indent=2));print(json.dumps(row),flush=True)
 print('INJECT_COMPLETE',flush=True)
if __name__=='__main__':main()
