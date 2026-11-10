import ctypes,json,statistics,os,gc
from pathlib import Path
import torch,torch_npu
P=Path(__file__).resolve().parent
lib=ctypes.CDLL(str(P/'libgemv.so'))
lib.fnx_raw_gemv.argtypes=[ctypes.c_void_p]*3+[ctypes.c_int]*4+[ctypes.c_void_p]
def native(x,w,blocks,batch):
 n,k=w.shape;y=torch.empty(((n+15)//16*16,),dtype=x.dtype,device=x.device)
 rc=lib.fnx_raw_gemv(x.data_ptr(),w.data_ptr(),y.data_ptr(),n,k,blocks,batch,torch.npu.current_stream().npu_stream)
 assert rc==0,(n,k,blocks,batch,rc)
 return y[:n].reshape(1,n)
def bench(g):
 samples=[]
 for _ in range(5):
  a=torch.npu.Event(enable_timing=True);b=torch.npu.Event(enable_timing=True);a.record()
  for _ in range(8):g.replay()
  b.record();b.synchronize();samples.append(a.elapsed_time(b)/8/128)
 return statistics.median(samples)
@torch.no_grad()
def main():
 torch.npu.set_device(0);torch.manual_seed(137);rows=[]
 for n,k in ((324,10240),(10240,320),(10240,2560),(6144,2560),(2560,6144),(1793,2560),(2560,640),(48,2560)):
  x=torch.randn(1,k,device='npu:0',dtype=torch.bfloat16)*.03;w=torch.randn(n,k,device=x.device,dtype=x.dtype)*.03
  pool_mib=int(os.environ.get('COLD_MIB','0'))
  pool_count=max(1,(pool_mib*1024*1024+w.numel()*2-1)//(w.numel()*2))
  ws=[w]+[w.clone() for _ in range(pool_count-1)]
  funcs={'baseline':lambda i:torch.nn.functional.linear(x,ws[i%pool_count])}
  for b in (20,40):
   for r in (1,4,8):
    if k*(10+r*2)+8192<=196608:funcs[f'v{b}r{r}']=lambda i,b=b,r=r:native(x,ws[i%pool_count],b,r)
  graphs={};outputs={}
  for name,fn in funcs.items():
   fn(0);torch.npu.synchronize();g=torch.npu.NPUGraph()
   with torch.npu.graph(g):
    for i in range(128):outputs[name]=fn(i)
   graphs[name]=g
  errors={name:dict(max_abs=0.,rel_l2=0.,exact=True) for name in funcs if name!='baseline'}
  for step in range(8):
   x.copy_(torch.randn_like(x)*(.03 if step%2 else .3))
   for g in graphs.values():g.replay()
   torch.npu.synchronize();ref=outputs['baseline'].float()
   for name,d in errors.items():
    diff=outputs[name].float()-ref;d['exact'] &= torch.equal(outputs[name],outputs['baseline']);d['max_abs']=max(d['max_abs'],diff.abs().max().item());d['rel_l2']=max(d['rel_l2'],(diff.norm()/ref.norm()).item())
  timings={name:bench(g) for name,g in graphs.items()};timings['baseline_repeat']=bench(graphs['baseline'])
  row=dict(N=n,K=k,pool_count=pool_count,timings_ms=timings,accuracy=errors);rows.append(row);(P/('gemv-probe-cold.json' if pool_mib else 'gemv-probe.json')).write_text(json.dumps(rows,indent=2));print(json.dumps(row),flush=True)
  del graphs,outputs,w,x,ws,funcs;gc.collect();torch.npu.empty_cache()
 print('COMPLETE',flush=True)
if __name__=='__main__':main()
