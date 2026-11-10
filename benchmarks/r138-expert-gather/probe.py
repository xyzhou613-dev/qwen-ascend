import ctypes,json,statistics,gc,os
from pathlib import Path
import torch,torch_npu
P=Path(__file__).resolve().parent
lib=ctypes.CDLL(str(P/'libgather.so'));lib.fnx_expert_gather.argtypes=[ctypes.c_void_p]*7+[ctypes.c_int]*5+[ctypes.c_void_p];lib.fnx_expert_gather.restype=ctypes.c_int
old=ctypes.CDLL(os.environ.get('FNX_EXPERT_GATHER_SO',str(P.parents[1]/'fnx/libfnx_expert_gather.so')));old.fnx_gather_pair.argtypes=[ctypes.c_void_p]*5+[ctypes.c_int]*3+[ctypes.c_void_p];old.fnx_gather_pair.restype=ctypes.c_int
def bench(g,calls=16):
 values=[]
 for _ in range(7):
  a=torch.npu.Event(enable_timing=True);b=torch.npu.Event(enable_timing=True);a.record()
  for _ in range(8):g.replay()
  b.record();b.synchronize();values.append(a.elapsed_time(b)/8/calls)
 return statistics.median(values)
@torch.no_grad()
def main():
 torch.npu.set_device(0);torch.manual_seed(138);E,K,N,G=512,2560,1280,40;d='npu:0'
 w=torch.randint(-(2**31),2**31-1,(E,K,N//8),device=d,dtype=torch.int32)
 s=torch.randn(E,G,N,device=d,dtype=torch.bfloat16)*.01
 ids=torch.randperm(E,device=d)[:10];x=torch.randn(1,K,device=d,dtype=torch.bfloat16)*.03
 tables={}
 for rows in (4,8):
  tables[rows]=tuple(torch.tensor([4*(e*r*columns+q*columns+c) for q in range(r) for e in range(10) for c in range(columns)],device=d,dtype=torch.int32) for r,columns in ((rows,N//8),(rows//4,N//2)))
 def run(name,up):
  if name=='baseline':
   ww=w.index_select(0,ids).permute(1,0,2).contiguous().view(K,-1);ss=s.index_select(0,ids).transpose(0,1).contiguous().view(G,-1)
  else:
   ww=torch.empty(K,10*N//8,device=d,dtype=torch.int32);ss=torch.empty(G,10*N,device=d,dtype=torch.bfloat16)
   if name=='previous':rc=old.fnx_gather_pair(w.data_ptr(),s.data_ptr(),ids.data_ptr(),ww.data_ptr(),ss.data_ptr(),K,G,N,torch.npu.current_stream().npu_stream)
   else:
    rows,overlap=map(int,name.split('_'));wi,si=tables[rows]
    rc=lib.fnx_expert_gather(w.data_ptr(),s.data_ptr(),ids.data_ptr(),wi.data_ptr(),si.data_ptr(),ww.data_ptr(),ss.data_ptr(),K,G,N,rows,overlap,torch.npu.current_stream().npu_stream)
   assert rc==0
  if up:return torch_npu.npu_weight_quant_batchmatmul(x,ww,ss,antiquant_group_size=64)
  return ww,ss
 results=[]
 for up in (False,True):
  graphs={};outputs={}
  for name in ('baseline','previous','4_0','4_1','8_0','8_1'):
   run(name,up);torch.npu.synchronize();g=torch.npu.NPUGraph()
   with torch.npu.graph(g):
    for _ in range(16):outputs[name]=run(name,up)
   graphs[name]=g
  exact={n:True for n in graphs if n!='baseline'}
  for step in range(32):
   ids.copy_(torch.randperm(E,device=d)[:10] if step%4 else torch.tensor([0,511,0,123,12,510,50,100,300,399],device=d));x.copy_(torch.randn_like(x)*.03)
   for g in graphs.values():g.replay()
   torch.npu.synchronize()
   for n in exact:
    a=outputs[n] if not up else (outputs[n],);b=outputs['baseline'] if not up else (outputs['baseline'],)
    exact[n] &= all(torch.equal(aa.view(torch.int32 if aa.dtype==torch.int32 else torch.int16),bb.view(torch.int32 if bb.dtype==torch.int32 else torch.int16)) for aa,bb in zip(a,b))
  records=[dict(mode=n,ms=bench(graphs[n])) for n in ('baseline','previous','4_0','4_1','8_0','8_1','baseline')]
  row=dict(include_up_matmul=up,exact=exact,records=records,dynamic_steps=32);results.append(row);(P/'probe.json').write_text(json.dumps(results,indent=2));print(json.dumps(row),flush=True)
  del graphs,outputs;gc.collect();torch.npu.empty_cache()
 print('GATHER_COMPLETE',flush=True)
if __name__=='__main__':main()
