import os
import ctypes,json,statistics,gc
from pathlib import Path
import torch,torch_npu
P=Path(__file__).resolve().parent;lib=ctypes.CDLL(os.environ.get('FNX_POST_SO',str(P.parents[1]/'fnx/libfnx_post.so')));lib.fnx_moe_merge.argtypes=[ctypes.c_void_p]*4+[ctypes.c_int,ctypes.c_void_p];lib.fnx_moe_merge.restype=ctypes.c_int
def native(e,s,g):
 y=torch.empty_like(e);rc=lib.fnx_moe_merge(e.data_ptr(),s.data_ptr(),g.data_ptr(),y.data_ptr(),e.shape[0],torch.npu.current_stream().npu_stream);assert rc==0;return y
def reference(e,s,g):return e+s*torch.sigmoid(g)
def diff(a,b):return dict(exact=torch.equal(a.view(torch.int16),b.view(torch.int16)),different=int((a.view(torch.int16)!=b.view(torch.int16)).sum().item()),max_abs=(a.float()-b.float()).abs().max().item())
def bench(g):
 v=[]
 for _ in range(7):
  a=torch.npu.Event(enable_timing=True);b=torch.npu.Event(enable_timing=True);a.record()
  for _ in range(8):g.replay()
  b.record();b.synchronize();v.append(a.elapsed_time(b)/8/256)
 return statistics.median(v)
@torch.no_grad()
def main():
 torch.npu.set_device(0);torch.manual_seed(140142);rows=[]
 bits=torch.arange(65536,dtype=torch.int32);bits=bits[(bits&0x7f80)!=0x7f80].to(torch.int16);g=bits.view(torch.bfloat16).to('npu:0').view(-1,1)
 e=torch.randn(g.shape[0],2560,device=g.device,dtype=g.dtype)*.03;s=torch.randn_like(e)*.03
 a,b=reference(e,s,g),native(e,s,g);torch.npu.synchronize();rows.append(dict(kind='finite_bf16_gate',**diff(a,b)));print(rows[-1],flush=True)
 del a,b,e,s,g;gc.collect();torch.npu.empty_cache()
 e=torch.randn(1,2560,device='npu:0',dtype=torch.bfloat16)*.03;s=torch.randn_like(e)*.03;g=torch.randn(1,1,device=e.device,dtype=e.dtype);graphs={};outputs={}
 for name,fn in (('baseline',reference),('native',native)):
  fn(e,s,g);torch.npu.synchronize();graph=torch.npu.NPUGraph()
  with torch.npu.graph(graph):
   for _ in range(256):outputs[name]=fn(e,s,g)
  graphs[name]=graph
 for step in range(32):
  e.copy_(torch.randn_like(e)*10.**((step%7)-3));s.copy_(torch.randn_like(s)*10.**((step%7)-3));g.copy_(torch.randn_like(g)*10.**((step%5)-2))
  for graph in graphs.values():graph.replay()
  torch.npu.synchronize();rows.append(dict(kind='dynamic',step=step,**diff(outputs['baseline'],outputs['native'])))
 for name,graph in graphs.items():rows.append(dict(kind='timing',mode=name,ms=bench(graph)));print(rows[-1],flush=True)
 (P/'moe-probe.json').write_text(json.dumps(rows,indent=2));print('MOE_PROBE_COMPLETE',flush=True)
if __name__=='__main__':main()
