import os
import ctypes,json,statistics
from pathlib import Path
import torch,torch_npu
P=Path(__file__).resolve().parent;lib=ctypes.CDLL(os.environ.get('FNX_POST_SO',str(P.parents[1]/'fnx/libfnx_post.so')));lib.fnx_post_mul.argtypes=[ctypes.c_void_p]*3+[ctypes.c_int]*2+[ctypes.c_void_p];lib.fnx_post_mul.restype=ctypes.c_int
def native(x,w,mode):
 y=torch.empty_like(x,dtype=torch.bfloat16);rc=lib.fnx_post_mul(x.data_ptr(),w.data_ptr(),y.data_ptr(),x.numel(),mode,torch.npu.current_stream().npu_stream);assert rc==0;return y
def reference(x,w,mode):return ((x.to(torch.bfloat16).float() if mode else x)*w).to(torch.bfloat16)
def bench(g):
 s=[]
 for _ in range(7):
  a=torch.npu.Event(enable_timing=True);b=torch.npu.Event(enable_timing=True);a.record()
  for _ in range(8):g.replay()
  b.record();b.synchronize();s.append(a.elapsed_time(b)/8/256)
 return statistics.median(s)
@torch.no_grad()
def main():
 torch.npu.set_device(0);torch.manual_seed(140141);rows=[]
 for n,mode in ((10240,0),(6144,1)):
  x=torch.randn(n,device='npu:0',dtype=torch.float32);w=torch.randn_like(x);graphs={};outputs={}
  for name,fn in (('baseline',reference),('native',native)):
   fn(x,w,mode);torch.npu.synchronize();g=torch.npu.NPUGraph()
   with torch.npu.graph(g):
    for _ in range(256):outputs[name]=fn(x,w,mode)
   graphs[name]=g
  exact=True;different=0
  for step in range(64):
   x.copy_(torch.randn_like(x)*10.**((step%13)-6));w.copy_(torch.randn_like(w)*10.**((step%9)-4))
   for g in graphs.values():g.replay()
   torch.npu.synchronize();different+=int((outputs['baseline'].view(torch.int16)!=outputs['native'].view(torch.int16)).sum().item());exact &= torch.equal(outputs['baseline'].view(torch.int16),outputs['native'].view(torch.int16))
  row=dict(n=n,mode=mode,dynamic_steps=64,exact=exact,different=different,timings_ms={name:bench(g) for name,g in graphs.items()});rows.append(row);print(json.dumps(row),flush=True)
 (P/'post-probe.json').write_text(json.dumps(rows,indent=2));print('POST_PROBE_COMPLETE',flush=True)
if __name__=='__main__':main()
