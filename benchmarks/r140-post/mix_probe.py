import os
import ctypes,json,statistics
from pathlib import Path
import torch,torch_npu
P=Path(__file__).resolve().parent
lib=ctypes.CDLL(os.environ.get('FNX_POST_SO',str(P.parents[1]/'fnx/libfnx_post.so')));lib.fnx_hc_sigmoid_reduce.argtypes=[ctypes.c_void_p]*3+[ctypes.c_int,ctypes.c_void_p];lib.fnx_hc_sigmoid_reduce.restype=ctypes.c_int
old=ctypes.CDLL(os.environ.get('FNX_REDUCE_SO',str(P.parents[1]/'fnx/libfnx_reduce.so')));old.fnx_weighted_reduce.argtypes=[ctypes.c_void_p]*4+[ctypes.c_int]*6+[ctypes.c_void_p];old.fnx_weighted_reduce.restype=ctypes.c_int
def reference(x,l):
 w=torch.sigmoid(l);b=x.shape[0];y=torch.empty(b,2560,device=x.device,dtype=x.dtype)
 rc=old.fnx_weighted_reduce(x.data_ptr(),w.data_ptr(),0,y.data_ptr(),b,4,2560,2560,0,0,torch.npu.current_stream().npu_stream);assert rc==0;return y
def native(x,l):
 y=torch.empty(x.shape[0],2560,device=x.device,dtype=x.dtype);rc=lib.fnx_hc_sigmoid_reduce(x.data_ptr(),l.data_ptr(),y.data_ptr(),x.shape[0],torch.npu.current_stream().npu_stream);assert rc==0;return y
def diff(a,b):return dict(exact=torch.equal(a.view(torch.int16),b.view(torch.int16)),different=int((a.view(torch.int16)!=b.view(torch.int16)).sum().item()),max_abs=(a.float()-b.float()).abs().max().item())
def bench(g):
 s=[]
 for _ in range(7):
  a=torch.npu.Event(enable_timing=True);b=torch.npu.Event(enable_timing=True);a.record()
  for _ in range(8):g.replay()
  b.record();b.synchronize();s.append(a.elapsed_time(b)/8/256)
 return statistics.median(s)
@torch.no_grad()
def main():
 torch.npu.set_device(0);torch.manual_seed(140140);rows=[]
 bits=torch.arange(65536,dtype=torch.int32);bits=bits[(bits&0x7f80)!=0x7f80].to(torch.int16);v=bits.view(torch.bfloat16).to('npu:0')
 l=torch.zeros(7,4,2560,device='npu:0',dtype=torch.bfloat16);l.view(-1)[:v.numel()]=v;x=torch.randn_like(l)*.03
 a,b=reference(x,l),native(x,l);torch.npu.synchronize();rows.append(dict(kind='finite_bf16_logits',**diff(a,b)));print(rows[-1],flush=True)
 x=torch.randn(1,4,2560,device='npu:0',dtype=torch.bfloat16)*.03;l=torch.randn_like(x)*.1;graphs={};out={}
 for name,fn in (('baseline',reference),('native',native)):
  fn(x,l);torch.npu.synchronize();g=torch.npu.NPUGraph()
  with torch.npu.graph(g):
   for _ in range(256):out[name]=fn(x,l)
  graphs[name]=g
 for step in range(32):
  x.copy_(torch.randn_like(x)*10.**((step%7)-3));l.copy_(torch.randn_like(l)*10.**((step%5)-2))
  for g in graphs.values():g.replay()
  torch.npu.synchronize();rows.append(dict(kind='dynamic',step=step,**diff(out['baseline'],out['native'])))
 for name,g in graphs.items():rows.append(dict(kind='timing',mode=name,ms=bench(g)));print(rows[-1],flush=True)
 (P/'mix-probe.json').write_text(json.dumps(rows,indent=2));print('MIX_PROBE_COMPLETE',flush=True)
if __name__=='__main__':main()
