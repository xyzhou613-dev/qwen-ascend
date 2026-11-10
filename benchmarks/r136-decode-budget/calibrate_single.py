import torch,torch_npu,ctypes,json,statistics,gc,os
from pathlib import Path
P=Path(__file__).resolve().parent
lib=ctypes.CDLL(str(P/'libnop.so'));lib.fnx_nop.argtypes=[ctypes.c_void_p,ctypes.c_int,ctypes.c_void_p]
def time_graph(g,calls):
 vals=[]
 for _ in range(5):
  a=torch.npu.Event(enable_timing=True);b=torch.npu.Event(enable_timing=True);a.record()
  for _ in range(5):g.replay()
  b.record();b.synchronize();vals.append(a.elapsed_time(b)/5/calls)
 return statistics.median(vals)
@torch.no_grad()
def main():
 rows=[]
 for device in range(torch.npu.device_count()):
  torch.npu.set_device(device);d=f'npu:{device}';out=torch.zeros(40,dtype=torch.int32,device=d);row=dict(device=device,visible=os.environ['ASCEND_RT_VISIBLE_DEVICES'],nop_us={})
  for blocks in (1,40):
   stream=torch.npu.current_stream().npu_stream;g=torch.npu.NPUGraph()
   with torch.npu.graph(g):
    for _ in range(1024):lib.fnx_nop(out.data_ptr(),blocks,stream)
   row['nop_us'][str(blocks)]=time_graph(g,1024)*1000;del g
  n=256*1024*1024//2;x=torch.ones(n,dtype=torch.bfloat16,device=d);y=torch.empty_like(x);g=torch.npu.NPUGraph()
  with torch.npu.graph(g):
   for _ in range(20):y.copy_(x);x.copy_(y)
  ms=time_graph(g,40);row.update(copy_ms=ms,stream_copy_GBps=(2*n*2)/(ms/1000)/1e9);del g,x,y
  x=torch.randn((4096,4096),dtype=torch.bfloat16,device=d);y=torch.randn_like(x);z=torch.empty_like(x);g=torch.npu.NPUGraph()
  with torch.npu.graph(g):
   for _ in range(20):torch.mm(x,y,out=z)
  ms=time_graph(g,20);row.update(gemm4096_ms=ms,gemm4096_TFLOPs=(2*4096**3)/(ms/1000)/1e12)
  rows.append(row);(P/('calibration-'+os.environ['ASCEND_RT_VISIBLE_DEVICES']+'.json')).write_text(json.dumps(rows,indent=2));print('CALIBRATION',row,flush=True)
  del g,x,y,z,out;gc.collect();torch.npu.empty_cache()
if __name__=='__main__':main()
