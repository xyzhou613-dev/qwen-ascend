import gc,json,statistics,sys,time,urllib.request
from pathlib import Path
import torch,torch_npu
from fnx import state_update as N
P=Path(__file__).resolve().parent;selector=int(sys.argv[1]);start=time.monotonic()
torch.set_num_threads(1);torch.npu.set_device(0);torch.npu.set_per_process_memory_fraction(1/128,0);torch.manual_seed(1510+selector)
def guard():
 h=json.load(urllib.request.urlopen('http://127.0.0.1:9002/health',timeout=5));assert h['service_revision']=='r151' and h['worker_alive'];assert torch.npu.mem_get_info()[0]>1536*1024**2 and time.monotonic()-start<220
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
guard()
results={}
for count in (1,8):
 inputs=[(torch.randn(48,128,128,device='npu')*.01,torch.rand(48,1,1,device='npu'),torch.randn(48,128,128,device='npu')*.01) for _ in range(count)]
 def updates(mode):
  for i in range(256):
   s,d,a=inputs[i%count];last=s*d+a if mode is None else N.run(s,d,a,mode)
  return last
 fns={'torch':lambda:updates(None)}
 for mode in (0,2,3,4,5,6):fns[f'native{mode}']=lambda mode=mode:updates(mode)
 result=bench(fns,2);result['per_update_us']={k:v/256 for k,v in result['median_us'].items()};results[str(count)]=result
 print('LONG_GRAPH_STATE_RESULT',selector,count,result['per_update_us'],flush=True)
 del inputs,fns;gc.collect();torch.npu.empty_cache()
(P/f'long-graph-device{selector}.json').write_text(json.dumps(dict(status='passed',selector=selector,updates_per_graph=256,results=results,exclusive=False),indent=2))
