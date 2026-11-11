import os
import ctypes,json,statistics,gc
from pathlib import Path
import torch,torch_npu
P=Path(__file__).resolve().parent
lib=ctypes.CDLL(os.environ.get('FNX_POST_SO',str(P.parents[1]/'fnx/libfnx_post.so')));lib.fnx_hc_activation.argtypes=[ctypes.c_void_p]*3+[ctypes.c_int]*2+[ctypes.c_void_p];lib.fnx_hc_activation.restype=ctypes.c_int
def native(x,mode):
 b=x.numel()//324;m=torch.empty(b,320,device=x.device,dtype=x.dtype);g=torch.empty(b,4,device=x.device,dtype=x.dtype)
 rc=lib.fnx_hc_activation(x.data_ptr(),m.data_ptr(),g.data_ptr(),b,mode,torch.npu.current_stream().npu_stream);assert rc==0;return m,g
def reference(x):return torch.nn.functional.silu(x[:,:320]/4),2*torch.sigmoid(x[:,320:]/4)
def difference(a,b):return dict(exact=torch.equal(a.view(torch.int16),b.view(torch.int16)),different=int((a.view(torch.int16)!=b.view(torch.int16)).sum().item()),max_abs=(a.float()-b.float()).abs().max().item())
def bench(g):
 samples=[]
 for _ in range(7):
  a=torch.npu.Event(enable_timing=True);b=torch.npu.Event(enable_timing=True);a.record()
  for _ in range(8):g.replay()
  b.record();b.synchronize();samples.append(a.elapsed_time(b)/8/512)
 return statistics.median(samples)
@torch.no_grad()
def main():
 torch.npu.set_device(0);torch.manual_seed(140);rows=[]
 bits=torch.arange(65536,dtype=torch.int32);bits=bits[(bits&0x7f80)!=0x7f80].to(torch.int16);values=bits.view(torch.bfloat16).to('npu:0');assert values.numel()==65280
 for kind in ('silu','gate'):
  b=204 if kind=='silu' else 16320;x=torch.zeros(b,324,device='npu:0',dtype=torch.bfloat16)
  if kind=='silu':x[:,:320]=values.view(b,320)
  else:x[:,320:]=values.view(b,4)
  expected=reference(x)
  for mode in range(4):
   out=native(x,mode);torch.npu.synchronize();row=dict(kind=kind,mode=mode,mixed=difference(expected[0],out[0]),gate=difference(expected[1],out[1]));rows.append(row);print(json.dumps(row),flush=True)
  del x,expected,out;gc.collect();torch.npu.empty_cache()
 x=torch.randn(1,324,device='npu:0',dtype=torch.bfloat16)*.1;graphs={};outputs={}
 for mode in (-1,0,1,2,3):
  fn=lambda:reference(x) if mode==-1 else native(x,mode);fn();torch.npu.synchronize();g=torch.npu.NPUGraph()
  with torch.npu.graph(g):
   for _ in range(512):outputs[mode]=fn()
  graphs[mode]=g
 for step in range(32):
  x.copy_(torch.randn_like(x)*10.**((step%7)-3))
  for g in graphs.values():g.replay()
  torch.npu.synchronize()
  for mode in (0,1,2,3):rows.append(dict(kind='dynamic',step=step,mode=mode,mixed=difference(outputs[-1][0],outputs[mode][0]),gate=difference(outputs[-1][1],outputs[mode][1])))
 for mode in (-1,0,1,2,3):row=dict(kind='timing',mode=mode,ms=bench(graphs[mode]));rows.append(row);print(json.dumps(row),flush=True)
 (P/'probe.json').write_text(json.dumps(rows,indent=2));print('ACTIVATION_PROBE_COMPLETE',flush=True)
if __name__=='__main__':main()
