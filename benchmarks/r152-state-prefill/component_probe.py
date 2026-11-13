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
guard();cases=[]
for i in range(64):
 guard();scale=2.**([-149,-130,-120,-60,0,60,100][i%7]);s=torch.randn(48,128,128,device='npu')*scale;a=torch.randn_like(s)*scale;d=torch.rand(48,1,1,device='npu')
 if i==0:s.fill_(1.+2.**-23);d.fill_(1.-2.**-23);a.fill_(-1.)
 if i%8==1:d.zero_()
 if i%8==2:d.fill_(1.)
 if i%8==3:a.copy_(-s*d)
 expected=s*d+a
 for mode in range(7):
  actual=N.run(s,d,a,mode);assert actual is not None
  assert eq(expected,actual),('STATE_MISMATCH',i,mode)
 cases.append(dict(case=i,modes=list(range(7)),exact=True));time.sleep(.01)
del s,a,d,expected,actual;gc.collect();torch.npu.empty_cache();guard()
inputs=[(torch.randn(48,128,128,device='npu')*.01,torch.rand(48,1,1,device='npu'),torch.randn(48,128,128,device='npu')*.01) for _ in range(8)]
fns={'torch':lambda:[s*d+a for s,d,a in inputs]}
for mode in range(7):fns[f'native{mode}']=lambda mode=mode:[N.run(s,d,a,mode) for s,d,a in inputs]
result=bench(fns,4)
r=dict(status='passed',container_selector=selector,server_physical_device=3 if selector==0 else 5,cases=cases,state_eight_updates=result,max_allocated_bytes=torch.npu.max_memory_allocated(),elapsed_s=time.monotonic()-start,exclusive=False)
(P/f'component-device{selector}.json').write_text(json.dumps(r,indent=2));print('STATE_COMPONENT_COMPLETE',selector,result['median_us'],flush=True)
