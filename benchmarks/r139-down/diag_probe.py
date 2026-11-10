import ctypes,json,statistics,gc,os
from pathlib import Path
import torch,torch_npu
P=Path(__file__).resolve().parent
lib=ctypes.CDLL(os.environ.get('FNX_EXPERT_GATHER_SO',str(P.parents[1]/'fnx/libfnx_expert_gather.so')));lib.fnx_gather_pair.argtypes=[ctypes.c_void_p]*5+[ctypes.c_int]*3+[ctypes.c_void_p];lib.fnx_gather_pair.restype=ctypes.c_int
def bench(g):
 samples=[]
 for _ in range(7):
  a=torch.npu.Event(enable_timing=True);b=torch.npu.Event(enable_timing=True);a.record()
  for _ in range(8):g.replay()
  b.record();b.synchronize();samples.append(a.elapsed_time(b)/8/16)
 return statistics.median(samples)
@torch.no_grad()
def main():
 torch.npu.set_device(0);torch.manual_seed(139);E,K,N,G=512,640,2560,10;d='npu:0'
 w=torch.randint(-(2**31),2**31-1,(E,K,N//8),device=d,dtype=torch.int32);s=torch.randn(E,G,N,device=d,dtype=torch.bfloat16)*.01
 x=torch.randn(10,K,device=d,dtype=torch.bfloat16)*.03;ids=torch.randperm(E,device=d)[:10];ends=torch.arange(1,11,device=d,dtype=torch.int64)
 diag=(torch.arange(10,device=d,dtype=torch.int64)*(10*N+N))[:,None]+torch.arange(N,device=d,dtype=torch.int64)[None,:]
 def run(mode):
  if mode=='gmm':return torch_npu.npu_grouped_matmul([x],[w.index_select(0,ids)],antiquant_scale=[s.index_select(0,ids)],group_list=ends,split_item=3,group_type=0,output_dtype=torch.bfloat16)[0]
  if mode=='native_diag':
   ww=torch.empty(K,10*N//8,device=d,dtype=torch.int32);ss=torch.empty(G,10*N,device=d,dtype=torch.bfloat16)
   rc=lib.fnx_gather_pair(w.data_ptr(),s.data_ptr(),ids.data_ptr(),ww.data_ptr(),ss.data_ptr(),K,G,N,torch.npu.current_stream().npu_stream);assert rc==0
  else:ww=w.index_select(0,ids).permute(1,0,2).contiguous().view(K,-1);ss=s.index_select(0,ids).transpose(0,1).contiguous().view(G,-1)
  y=torch_npu.npu_weight_quant_batchmatmul(x,ww,ss,antiquant_group_size=64)
  return y.view(-1).index_select(0,diag.view(-1)).view(10,N)
 results=[]
 for mode in ('gmm','diag','native_diag','gmm'):
  run(mode);torch.npu.synchronize();g=torch.npu.NPUGraph()
  with torch.npu.graph(g):
   for _ in range(16):out=run(mode)
  exact=True;max_abs=0.;rel_l2=0.
  for step in range(32):
   x.copy_(torch.randn_like(x)*.03);ids.copy_(torch.randperm(E,device=d)[:10]);expected=run('gmm');g.replay();torch.npu.synchronize()
   exact &= torch.equal(expected.view(torch.int16),out.view(torch.int16));max_abs=max(max_abs,(out.float()-expected.float()).abs().max().item());rel_l2=max(rel_l2,((out.float()-expected.float()).norm()/expected.float().norm()).item())
  row=dict(mode=mode,exact=exact,max_abs=max_abs,rel_l2=rel_l2,ms=bench(g));results.append(row);(P/'diag-probe.json').write_text(json.dumps(results,indent=2));print(json.dumps(row),flush=True)
  del g,out;gc.collect();torch.npu.empty_cache()
 print('DIAG_COMPLETE',flush=True)
if __name__=='__main__':main()
