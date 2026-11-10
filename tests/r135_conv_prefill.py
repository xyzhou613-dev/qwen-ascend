import ctypes,json,statistics,os
from pathlib import Path
import torch,torch_npu
OUT=Path(os.environ.get('FNX_BENCH_OUT','/models/fnx-diagnostics/r135-conv-check'));OUT.mkdir(parents=True,exist_ok=True)
lib=ctypes.CDLL(os.environ.get('FNX_CONV_PREFILL_SO',str(Path(__file__).resolve().parents[1]/'fnx/libfnx_conv_prefill.so')));lib.fnx_conv_prefill.argtypes=[ctypes.c_void_p]*3+[ctypes.c_int]*2+[ctypes.c_void_p];lib.fnx_conv_prefill.restype=ctypes.c_int
@torch.no_grad()
def main():
 torch.npu.set_device(0);torch.manual_seed(135135);records=[]
 for D,T in ((128,65),(128,124),(128,129),(10240,128),(10240,512),(10240,2048),(10240,8192)):
  x=torch.randn((D,T+3),device='npu:0',dtype=torch.bfloat16)*.1;w=torch.randn((D,4),device='npu:0',dtype=torch.bfloat16)*.1
  def ref():return torch.stack([w[:,j:j+1]*x[:,j:j+T] for j in range(4)],dim=-1).sum(-1)
  def native():
   y=torch.empty((D,T),device=x.device,dtype=x.dtype)
   rc=lib.fnx_conv_prefill(x.data_ptr(),w.data_ptr(),y.data_ptr(),D,T,torch.npu.current_stream().npu_stream)
   assert rc==0,rc
   return y
  graphs={};outputs={}
  for name,f in (('baseline',ref),('native',native)):
   f();torch.npu.synchronize();g=torch.npu.NPUGraph()
   with torch.npu.graph(g):outputs[name]=f()
   graphs[name]=g
  exact=True;max_abs=0;count=0
  for _ in range(8):
   x.copy_(torch.randn_like(x)*.1);w.copy_(torch.randn_like(w)*.1)
   for g in graphs.values():g.replay()
   torch.npu.synchronize();a,b=outputs['baseline'],outputs['native']
   exact &= torch.equal(a.view(torch.int16),b.view(torch.int16));count+=int((a.view(torch.int16)!=b.view(torch.int16)).sum().item())
   max_abs=max(max_abs,(a.float()-b.float()).abs().max().item())
  timings=[]
  for name in ('baseline','native','baseline','native'):
   samples=[]
   for _ in range(5):
    a=torch.npu.Event(enable_timing=True);b=torch.npu.Event(enable_timing=True);a.record()
    for _ in range(8):graphs[name].replay()
    b.record();b.synchronize();samples.append(a.elapsed_time(b)/8)
   timings.append(dict(mode=name,ms=statistics.median(samples)))
  row=dict(D=D,T=T,exact=exact,mismatched=count,max_abs=max_abs,timings=timings);records.append(row)
  (OUT/'probe.json').write_text(json.dumps(records,indent=2));print('CONV_PREFILL',json.dumps(row),flush=True)
  del graphs,outputs;torch.npu.empty_cache()
 print('CONV_PREFILL_COMPLETE',flush=True)
if __name__=='__main__':main()
