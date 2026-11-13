import gc,json,statistics,sys,time,urllib.request
from pathlib import Path
import torch,torch_npu
sys.path.insert(0,str(Path(__file__).resolve().parent/'candidate'))
from fnx import state_update as N,engine as E
import os
P=Path(__file__).resolve().parent;selector=int(sys.argv[1]);start=time.monotonic()
torch.set_num_threads(1);torch.npu.set_device(0);torch.npu.set_per_process_memory_fraction(1/64,0);torch.manual_seed(1510+selector)
def guard():
 h=json.load(urllib.request.urlopen('http://127.0.0.1:9002/health',timeout=5));assert h['service_revision']=='r151' and h['worker_alive'];assert torch.npu.mem_get_info()[0]>2560*1024**2 and time.monotonic()-start<220
def eq(a,b):return torch.equal(a.cpu().contiguous().view(torch.uint8),b.cpu().contiguous().view(torch.uint8))
def bench(functions,loops):
 stream=torch.npu.Stream();graphs={};outputs={}
 with torch.npu.stream(stream):
  for name,fn in functions.items():
   fn();stream.synchronize();g=torch.npu.NPUGraph()
   with torch.npu.graph(g,stream=stream):out=fn()
   graphs[name]=g;outputs[name]=out;g.replay();stream.synchronize()
  times={name:[] for name in graphs}
  for iteration in range(16):
   guard()
   for name in (list(graphs) if iteration%2==0 else list(reversed(graphs))):
    a=torch.npu.Event(enable_timing=True);b=torch.npu.Event(enable_timing=True);a.record(stream)
    for _ in range(loops):graphs[name].replay()
    b.record(stream);b.synchronize();times[name].append(a.elapsed_time(b)*1000/loops);time.sleep(.02)
 for g in graphs.values():g.reset()
 return dict(times_us=times,median_us={k:statistics.median(v) for k,v in times.items()})
guard();original=N.run;kernel_mode=4;calls=0
def run(*args,**kwargs):
 global calls
 out=original(*args,mode=kernel_mode,**kwargs);assert out is not None;calls+=1;return out
N.run=run
records=[]
for T in (512,):
 q=torch.randn(T,48,128,dtype=torch.bfloat16,device='npu')*.03;k=torch.randn_like(q)*.03;v=torch.randn_like(q)*.03
 log_g=-torch.rand(T,48,device='npu')*.02;beta=torch.rand(T,48,1,device='npu',dtype=torch.bfloat16);state=torch.randn(1,48,128,128,device='npu')*.01
 def function(flag,mode,inplace=0):
  global kernel_mode
  kernel_mode=mode;os.environ['FNX_STATE_UPDATE']=str(flag);os.environ['FNX_STATE_INPLACE']=str(inplace)
  return E.delta_chunk_fp32(q,k,v,log_g,beta,state)
 before=state.clone();reference=function(0,4);torch.npu.synchronize()
 for mode in (0,2,3,4):
  candidate=function(1,mode,1);assert all(eq(a,b) for a,b in zip(reference,candidate)),('FUNCTION_MISMATCH',T,mode)
 assert eq(state,before),('INPUT_STATE_MUTATED',T)
 del reference,candidate,before;gc.collect();torch.npu.empty_cache()
 fns={'torch':lambda:function(0,4)}
 fns['native4']=lambda:function(1,4,0)
 fns['inplace4']=lambda:function(1,4,1)
 timing=bench(fns,2);records.append(dict(T=T,timing=timing,outputs_and_fp32_state_exact=True));print('DELTA_FUNCTION_RESULT',selector,T,timing['median_us'],flush=True)
 del q,k,v,log_g,beta,state,fns;gc.collect();torch.npu.empty_cache()
(P/f'delta-function-device{selector}.json').write_text(json.dumps(dict(status='passed',selector=selector,records=records,native_calls=calls,max_allocated_bytes=torch.npu.max_memory_allocated(),exclusive=False),indent=2))
