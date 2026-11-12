import gc,json,statistics,sys,time,urllib.request
from pathlib import Path
import torch,torch_npu
from fnx import prefill_output as N
P=Path(__file__).resolve().parent;selector=int(sys.argv[1]);start=time.monotonic()
torch.set_num_threads(1);torch.npu.set_device(0);torch.npu.set_per_process_memory_fraction(1/128,0);torch.manual_seed(1510+selector)
def guard():
 h=json.load(urllib.request.urlopen('http://127.0.0.1:9002/health',timeout=5));assert h['service_revision']=='r150' and h['worker_alive'];assert torch.npu.mem_get_info()[0]>1536*1024**2 and time.monotonic()-start<220
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
 guard();scale=2.**([-60,-20,0,20,60][i%5]);l=torch.randn(48,64,128,device='npu')*scale;r=torch.randn_like(l)*scale
 if i%8==3:r.copy_(-l)
 if i%8==4:l.fill_(1.+2.**-8);r.fill_(2.**-24)
 reference=(l+r).to(torch.bfloat16).transpose(0,1).contiguous();output=torch.full((192,48,128),-13.,device='npu',dtype=torch.bfloat16)
 for mode in range(1,6):
  assert N.add_into(l,r,output,1,mode)
  assert eq(reference,output[64:128]),('LAYOUT_MISMATCH',i,mode)
  assert bool((output[:64]==-13.).all()) and bool((output[128:]==-13.).all()),('OFFSET_CORRUPTION',i,mode)
 cases.append(dict(case=i,modes=[1,2,3,4,5],exact=True,untouched_neighbors=True));time.sleep(.01)
del l,r,output,reference;gc.collect();torch.npu.empty_cache();guard()
inputs=[(torch.randn(48,64,128,device='npu')*.01,torch.randn(48,64,128,device='npu')*.01) for _ in range(32)]
output=torch.empty(2048,48,128,device='npu',dtype=torch.bfloat16)
def reference():return torch.cat([(l+r).transpose(0,1) for l,r in inputs],0).to(torch.bfloat16)
def direct(mode):
 for block,(l,r) in enumerate(inputs):assert N.add_into(l,r,output,block,mode)
 return output
fns={'torch':reference}
for mode in range(1,6):
 fns[f'direct{mode}']=lambda mode=mode:direct(mode)
 assert eq(reference(),direct(mode)),('ASSEMBLY_MISMATCH',mode)
result=bench(fns,2)
r=dict(status='passed',container_selector=selector,server_physical_device=3 if selector==0 else 5,cases=cases,output_2048_assembly=result,max_allocated_bytes=torch.npu.max_memory_allocated(),elapsed_s=time.monotonic()-start,exclusive=False)
(P/f'layout-device{selector}.json').write_text(json.dumps(r,indent=2));print('DIRECT_OUTPUT_COMPONENT_COMPLETE',selector,result['median_us'],flush=True)
