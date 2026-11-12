"""Online, memory-bounded operator checks. No Engine or service mutation."""
import ctypes,json,math,statistics,sys,time,urllib.request
from pathlib import Path
import torch,torch_npu
import torch.nn.functional as F
from fnx import norm_post,norm_pair
P=Path(__file__).resolve().parent;selector=int(sys.argv[1]);started=time.monotonic()
torch.set_num_threads(1);torch.npu.set_device(0);torch.npu.set_per_process_memory_fraction(1/128,0);torch.manual_seed(1490+selector)
def guard():
    h=json.load(urllib.request.urlopen('http://127.0.0.1:9002/health',timeout=5));assert h['service_revision']=='r144' and h['worker_alive']
    assert torch.npu.mem_get_info()[0]>1536*1024**2 and time.monotonic()-started<150
def bits(x):return x.detach().cpu().contiguous().view(torch.int16)
def same(a,b):return torch.equal(bits(a),bits(b))
def ref(y):return norm_post.run(y[:,:2048].view(1,16,128))/math.sqrt(128),norm_post.run(y[:,2048:4096].view(1,16,128))
def matches(a,b):return all(same(x,y) for x,y in zip(a,b))
guard();lib=norm_pair.library()
lib.fnx_norm_sum_trace.argtypes=[ctypes.c_void_p]*4+[ctypes.c_int,ctypes.c_void_p];lib.fnx_norm_sum_trace.restype=ctypes.c_int
lib.fnx_norm_scale_trace.argtypes=[ctypes.c_void_p]*3;lib.fnx_norm_scale_trace.restype=ctypes.c_int
traces=[];failure_cases=list(Path('/models/fnx-diagnostics/r148-restart-ab-20261112').glob('rounding-sparse-case-*.pt'))
for case in failure_cases:
    original=torch.load(case,map_location='cpu',weights_only=False);x=original['input'].npu();sq=x*x
    expected=sq.sum(-1,keepdim=True);float_sum=sq.float().sum(-1,keepdim=True)
    modes=[]
    for mode in (0,1):
        ps=torch.empty_like(x);fs=torch.empty(16,8,dtype=torch.float32,device='npu');bs=torch.empty(16,16,dtype=torch.bfloat16,device='npu')
        rc=lib.fnx_norm_sum_trace(x.data_ptr(),ps.data_ptr(),fs.data_ptr(),bs.data_ptr(),mode,torch.npu.current_stream().npu_stream);assert rc==0
        torch.npu.synchronize();assert same(ps,sq)
        modes.append(dict(mode=mode,square_bits_equal=True,native_fp32_sum=float(fs[0,0]),native_bf16_sum=float(bs[0,0]),native_bf16_sum_bits=int(bits(bs)[0,0])))
    y=torch.zeros(1,10240,dtype=torch.bfloat16,device='npu');y[:,:2048].copy_(x.view(1,-1));y[:,2048:4096].copy_(x.view(1,-1))
    expected_output=ref(y)
    for joint in (False,True):assert matches(expected_output,norm_pair.run(y,joint_reduce=joint)),('SAVED_REPRODUCER',case.name,joint)
    row=dict(case=case.name,reference_bf16_sum=float(expected[0,0,0]),reference_bf16_sum_bits=int(bits(expected)[0,0,0]),torch_fp32_sum=float(float_sum[0,0,0]),exact_real_sum=float(sq.cpu().double()[0,0].sum()),native_modes=modes,partial_split_exact=True,partial_joint_exact=True)
    traces.append(row);print('SUM_TRACE',row,flush=True)
(P/f'sum-trace-device{selector}.json').write_text(json.dumps(traces,indent=2))
# Check the actual new C++ Muls/Cast sequence, not just a Torch approximation.
enc=torch.arange(65536,dtype=torch.int32).to(torch.int16).view(torch.bfloat16).npu();scaled=torch.empty_like(enc)
assert lib.fnx_norm_scale_trace(enc.data_ptr(),scaled.data_ptr(),torch.npu.current_stream().npu_stream)==0
scale_different=int((bits(enc/math.sqrt(128))!=bits(scaled)).sum());assert scale_different==0,('SCALE_ENCODINGS',scale_different)
cases=[]
for i in range(512):
    if i%32==0:guard()
    y=torch.randn(1,10240,dtype=torch.bfloat16,device='npu')*[.003,.03,.3,1.,3.,10.,30.,100.][i%8]
    if i%3==0:y=F.silu(y)
    if i%16==15:y[:,:4096].copy_(torch.randint(-32768,32767,(1,4096),dtype=torch.int16).view(torch.bfloat16).npu())
    if i%16==14:
        y.zero_();y[:,80]=1.;y[:,61]=.0625;y[:,60]=2.**-12;y[:,99]=2.**-12;y[:,2048:4096].copy_(y[:,:2048])
    expected=ref(y)
    row=dict(case=i,split_exact=matches(expected,norm_pair.run(y)),joint_exact=matches(expected,norm_pair.run(y,joint_reduce=True)))
    assert row['split_exact'] and row['joint_exact'],row
    cases.append(row);time.sleep(.002)
inputs=[torch.randn(1,10240,dtype=torch.bfloat16,device='npu')*.03 for _ in range(32)]
stream=torch.npu.Stream();torch.npu.synchronize();graphs={};outputs={}
with torch.npu.stream(stream):
    for name,fn in [('r144',ref),('split',norm_pair.run),('joint',lambda y:norm_pair.run(y,joint_reduce=True))]:
        fn(inputs[0]);stream.synchronize();g=torch.npu.NPUGraph()
        with torch.npu.graph(g,stream=stream):out=[fn(y) for y in inputs]
        graphs[name]=g;outputs[name]=out;g.replay();stream.synchronize()
    for name in ('split','joint'):assert all(matches(a,b) for a,b in zip(outputs['r144'],outputs[name]))
    times={name:[] for name in graphs}
    for iteration in range(16):
        guard()
        for name in (list(graphs) if iteration%2==0 else list(reversed(graphs))):
            a=torch.npu.Event(enable_timing=True);b=torch.npu.Event(enable_timing=True);a.record(stream)
            for _ in range(8):graphs[name].replay()
            b.record(stream);b.synchronize();times[name].append(a.elapsed_time(b)*1000/256);time.sleep(.035)
result=dict(container_selector=selector,server_physical_device=3 if selector==0 else 5,scale_encodings=65536,scale_differences=scale_different,cases=cases,saved_counterexamples=len(failure_cases),graph_cases=32,graph_exact=True,
    times_us=times,median_us={k:statistics.median(v) for k,v in times.items()},max_allocated_bytes=torch.npu.max_memory_allocated(),elapsed_s=time.monotonic()-started,exclusive=False)
(P/f'component-device{selector}.json').write_text(json.dumps(result,indent=2));print('PARTIAL_COMPONENT_COMPLETE',selector,result['median_us'],result['max_allocated_bytes'],flush=True)
for g in graphs.values():g.reset()
