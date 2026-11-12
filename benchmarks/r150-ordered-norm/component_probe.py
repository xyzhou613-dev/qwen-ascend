"""Fixed-layout CANN-ordered norm: saved failures, sparse boundaries, graphs."""
import ctypes,json,math,statistics,sys,time,urllib.request
from pathlib import Path
import torch,torch_npu
import torch.nn.functional as F
from fnx import norm_post,norm_ordered
P=Path(__file__).resolve().parent;selector=int(sys.argv[1]);started=time.monotonic()
torch.set_num_threads(1);torch.npu.set_device(0);torch.npu.set_per_process_memory_fraction(1/128,0);torch.manual_seed(1500+selector)
def guard():
    h=json.load(urllib.request.urlopen('http://127.0.0.1:9002/health',timeout=5));assert h['service_revision']=='r144' and h['worker_alive']
    assert torch.npu.mem_get_info()[0]>1536*1024**2 and time.monotonic()-started<190
def bits(x):return x.detach().cpu().contiguous().view(torch.int16)
def same(a,b):return torch.equal(bits(a),bits(b))
def ref(y):return norm_post.run(y[:,:2048].view(1,16,128))/math.sqrt(128),norm_post.run(y[:,2048:4096].view(1,16,128))
def matches(a,b):return all(same(x,y) for x,y in zip(a,b))
guard();lib=ctypes.CDLL(str(P/'fnx/libfnx_norm_ordered.so'))
lib.fnx_norm_ordered.argtypes=[ctypes.c_void_p]*3+[ctypes.c_int]*3+[ctypes.c_void_p];lib.fnx_norm_ordered.restype=ctypes.c_int
repro=[]
for file in sorted(Path('/models/fnx-diagnostics/r148-restart-ab-20261112').glob('rounding-sparse-case-*.pt')):
    data=torch.load(file,map_location='cpu',weights_only=False);x=data['input'].npu();out=torch.empty_like(x)
    assert lib.fnx_norm_ordered(x.data_ptr(),norm_post.table(x.device).data_ptr(),out.data_ptr(),16,2048,0,torch.npu.current_stream().npu_stream)==0
    assert same(norm_post.run(x),out),('ORIGINAL_COUNTEREXAMPLE',file.name)
    repro.append(dict(case=file.name,original_shape_exact=True,query_scaling=False))
cases=[]
for i in range(4608):
    if i%32==0:guard()
    if i<512:
        y=torch.randn(1,10240,dtype=torch.bfloat16,device='npu')*[.003,.03,.3,1.,3.,10.,30.,100.][i%8]
        if i%3==0:y=F.silu(y)
        if i%16==15:y[:,:4096].copy_(torch.randint(-32768,32767,(1,4096),dtype=torch.int16).view(torch.bfloat16).npu())
        kind='random_silu_encodings'
    else:
        j=i-512;positions=torch.rand(32,128).argsort(-1);host=torch.zeros(32,128,dtype=torch.bfloat16)
        # Sum near a BF16 half-ulp, with tiny FP32 contributions whose
        # placement exposes different reduction trees.
        host.scatter_(-1,positions[:,:1],1.)
        host.scatter_(-1,positions[:,1:2],2.**(-4-(j//512)%3))
        host.scatter_(-1,positions[:,2:3+j%16],2.**(-10-(j//16)%8))
        y=torch.zeros(1,10240,dtype=torch.bfloat16,device='npu');y[:,:4096].copy_(host.reshape(1,-1).npu());kind='sparse_rounding_boundaries'
    expected=ref(y);actual=norm_ordered.run(y);equal=matches(expected,actual)
    row=dict(case=i,kind=kind,bitwise_equal=equal);cases.append(row)
    if not equal:
        torch.save(dict(input=y.cpu(),reference=[x.cpu() for x in expected],candidate=[x.cpu() for x in actual]),P/f'rejected-device{selector}-case{i}.pt')
        (P/f'component-device{selector}.json').write_text(json.dumps(dict(status='rejected',cases=cases),indent=2));raise AssertionError(('ORDERED_MISMATCH',selector,i,kind))
    if i%512==0:print('ORDERED_ACCURACY_PASS',selector,i,flush=True)
    time.sleep(.001)
inputs=[torch.randn(1,10240,dtype=torch.bfloat16,device='npu')*.03 for _ in range(32)]
stream=torch.npu.Stream();torch.npu.synchronize();graphs={};outputs={}
with torch.npu.stream(stream):
    for name,fn in [('r144',ref),('ordered',norm_ordered.run)]:
        fn(inputs[0]);stream.synchronize();g=torch.npu.NPUGraph()
        with torch.npu.graph(g,stream=stream):out=[fn(y) for y in inputs]
        graphs[name]=g;outputs[name]=out;g.replay();stream.synchronize()
    assert all(matches(a,b) for a,b in zip(outputs['r144'],outputs['ordered']))
    times={name:[] for name in graphs}
    for iteration in range(16):
        guard()
        for name in (list(graphs) if iteration%2==0 else list(reversed(graphs))):
            a=torch.npu.Event(enable_timing=True);b=torch.npu.Event(enable_timing=True);a.record(stream)
            for _ in range(8):graphs[name].replay()
            b.record(stream);b.synchronize();times[name].append(a.elapsed_time(b)*1000/256);time.sleep(.035)
r=dict(status='passed',container_selector=selector,server_physical_device=3 if selector==0 else 5,counterexamples=repro,cases=cases,graph_exact=True,times_us=times,median_us={k:statistics.median(v) for k,v in times.items()},max_allocated_bytes=torch.npu.max_memory_allocated(),elapsed_s=time.monotonic()-started,exclusive=False)
(P/f'component-device{selector}.json').write_text(json.dumps(r,indent=2));print('ORDERED_COMPONENT_COMPLETE',selector,r['median_us'],flush=True)
for g in graphs.values():g.reset()
