import ctypes,json,statistics,gc
from pathlib import Path
import torch,torch_npu
P=Path(__file__).resolve().parent
lib=ctypes.CDLL(str(P/'libdown_gather.so'));lib.fnx_down_gather.argtypes=[ctypes.c_void_p]*6;lib.fnx_down_gather.restype=ctypes.c_int
def bench(g):
 samples=[]
 for _ in range(5):
  a=torch.npu.Event(enable_timing=True);b=torch.npu.Event(enable_timing=True);a.record()
  for _ in range(8):g.replay()
  b.record();b.synchronize();samples.append(a.elapsed_time(b)/8/16)
 return statistics.median(samples)
@torch.no_grad()
def main():
 torch.npu.set_device(0);torch.manual_seed(139);E,K,N,G=512,640,2560,10;d='npu:0'
 w=torch.randint(-(2**31),2**31-1,(E,K,N//8),device=d,dtype=torch.int32)
 s=torch.randn(E,G,N,device=d,dtype=torch.bfloat16)*.01;x=torch.randn(10,K,device=d,dtype=torch.bfloat16)*.03;ids=torch.randperm(E,device=d)[:10]
 ends=torch.arange(1,11,device=d,dtype=torch.int64);counts=torch.ones(10,device=d,dtype=torch.int64)
 def gather(native):
  if not native:return w.index_select(0,ids),s.index_select(0,ids)
  a=torch.empty(10,K,N//8,device=d,dtype=torch.int32);b=torch.empty(10,G,N,device=d,dtype=torch.bfloat16)
  rc=lib.fnx_down_gather(w.data_ptr(),s.data_ptr(),ids.data_ptr(),a.data_ptr(),b.data_ptr(),torch.npu.current_stream().npu_stream);assert rc==0;return a,b
 def run(name):
  native=name.startswith('native');ww,ss=gather(native)
  kind,tune,split=map(int,name.split('_')[1:]);kw=dict(group_list=counts if kind else ends,group_list_type=kind,split_item=split,group_type=0,output_dtype=torch.bfloat16)
  if tune>=0:kw['tuning_config']=[tune]
  return torch_npu.npu_grouped_matmul([x],[ww],antiquant_scale=[ss],**kw)[0]
 names=['base_0_-1_3','native_0_-1_3']+[f'base_{kind}_{t}_{split}' for kind in (0,1) for t in (1,2,4,8,10) for split in (2,3)]
 records=[];reference=None
 for name in names:
  try:
   run(name);torch.npu.synchronize();g=torch.npu.NPUGraph()
   with torch.npu.graph(g):
    for _ in range(16):out=run(name)
   exact=True;max_abs=0.
   for step in range(16):
    x.copy_(torch.randn_like(x)*.03);ids.copy_(torch.randperm(E,device=d)[:10] if step%4 else torch.tensor([0,511,0,123,12,510,50,100,300,399],device=d))
    expected=run('base_0_-1_3');g.replay();torch.npu.synchronize();exact &= torch.equal(expected.view(torch.int16),out.view(torch.int16));max_abs=max(max_abs,(expected.float()-out.float()).abs().max().item())
   row=dict(mode=name,exact=exact,max_abs=max_abs,ms=bench(g));del g,out;gc.collect();torch.npu.empty_cache()
  except Exception as e:row=dict(mode=name,error=str(e)[:350])
  records.append(row);(P/'probe.json').write_text(json.dumps(records,indent=2));print(json.dumps(row),flush=True)
 print('DOWN_PROBE_COMPLETE',flush=True)
if __name__=='__main__':main()
