"""Compare direct packed-W4 GEMV with production gather + W4A16 GEMM."""
import ctypes,gc,json,os,statistics,sys,time,urllib.request
from pathlib import Path
import torch,torch_npu
from safetensors import safe_open
from fnx import engine as E
P=Path(__file__).resolve().parent;selector=int(sys.argv[1]);started=time.monotonic()
torch.set_num_threads(1);torch.npu.set_device(0);torch.npu.set_per_process_memory_fraction(1/32,0)
torch.manual_seed(15480+selector)
lib=ctypes.CDLL(str(P/'libfnx_w4_gemv-tile32.so'))
lib.fnx_w4_gemv.argtypes=[ctypes.c_void_p]*5+[ctypes.c_int]*3+[ctypes.c_void_p]
lib.fnx_w4_gemv.restype=ctypes.c_int

def guard():
    h=json.load(urllib.request.urlopen('http://127.0.0.1:9002/health',timeout=5))
    assert h['service_revision']=='r152' and h['worker_alive']
    assert torch.npu.mem_get_info()[0]>512*1024**2
    assert time.monotonic()-started<650

def error(a,b):
    a=a.detach().cpu().float();b=b.detach().cpu().float()
    return dict(relative_l2=float((a-b).norm()/a.norm().clamp_min(1e-12)),
        max_abs=float((a-b).abs().max()),finite=bool(torch.isfinite(b).all()))

@torch.no_grad()
def main():
    cfg=json.load(open(P/'production-selected-config.json'));records=[]
    source=Path(cfg['FNX_W4A16_DIR']+'-em')/'L00.w4.safetensors'
    for name,per_expert in [('down_proj',1)]:
        guard();key='mlp.experts.'+name
        with safe_open(str(source),framework='pt',device='cpu') as f:
            obj=E.ExpW4(f.get_tensor(key+'.w4').npu(),f.get_tensor(key+'.s8t').npu(),
                        f.get_tensor(key+'.base').npu(),64,expert_major=True)
        with safe_open(str(P/'private-artifacts'/f'L00-{name}-rowpack256.safetensors'),framework='pt',device='cpu') as f:
            packed=f.get_tensor('packed').npu();scale=f.get_tensor('scale').npu()
        x=torch.randn(10 if per_expert else 1,obj.K,device='npu',dtype=torch.bfloat16)*.03
        ids=torch.arange(10,device='npu',dtype=torch.int64)
        def baseline(ix):
            return obj.apply_diag(x,ix) if per_expert else obj.apply(x,ix).view(10,obj.N)
        def direct(ix):
            y=torch.empty((10,obj.N),device='npu',dtype=torch.float32)
            rc=lib.fnx_w4_gemv(packed.data_ptr(),scale.data_ptr(),ix.data_ptr(),x.data_ptr(),y.data_ptr(),
                    obj.K,obj.N,per_expert,torch.npu.current_stream().npu_stream)
            assert rc==0,rc
            return y.to(torch.bfloat16)
        errors=[];stream=torch.npu.Stream()
        for sample in range(20):
            guard();ids.copy_(torch.randint(0,256,(10,),device='npu'))
            x.copy_(torch.randn_like(x)*[.003,.03,.3,1.][sample%4])
            e=error(baseline(ids),direct(ids));assert e['finite'] and e['relative_l2']<.03,e
            errors.append(e)
        with torch.npu.stream(stream):
            g=torch.npu.NPUGraph()
            with torch.npu.graph(g,stream=stream):dynamic=direct(ids)
            stream.synchronize()
            for _ in range(8):
                ids.copy_(torch.randint(0,256,(10,),device='npu'));x.copy_(torch.randn_like(x)*.03)
                eager=direct(ids);g.replay();stream.synchronize()
                assert torch.equal(eager.cpu(),dynamic.cpu()),'fresh-input graph replay mismatch'
            g.reset();del g,dynamic,eager
        banks=torch.stack([torch.remainder(torch.arange(10)+i*10,256) for i in range(64)]).to('npu')
        graphs={};outputs={};times={n:[] for n in ('baseline','direct')}
        torch.npu.synchronize();torch.npu.empty_cache()
        with torch.npu.stream(stream):
            for label,fn in [('baseline',baseline),('direct',direct)]:
                fn(banks[0]);stream.synchronize();g=torch.npu.NPUGraph()
                with torch.npu.graph(g,stream=stream,pool=graphs['baseline'].pool() if graphs else None):
                    for bank in banks:out=fn(bank)
                graphs[label]=g;outputs[label]=out;g.replay();stream.synchronize()
            for repeat in range(12):
                guard()
                for label in (list(graphs) if repeat%2==0 else list(reversed(graphs))):
                    a=torch.npu.Event(enable_timing=True);b=torch.npu.Event(enable_timing=True);a.record(stream)
                    for _ in range(2):graphs[label].replay()
                    b.record(stream);b.synchronize();times[label].append(a.elapsed_time(b)*1000/128)
                time.sleep(.02)
        record=dict(name=name,K=obj.K,N=obj.N,errors=errors,times_us=times,
            median_us={n:statistics.median(t) for n,t in times.items()},
            fresh_input_graph_verified=True,weight_subset=256,baseline_experts=512,
            includes_output_bf16_cast=True,exclusive=False)
        records.append(record);print('GEMV_RESULT',selector,name,record['median_us'],flush=True)
        (P/f'gemv-tile32-device{selector}.json').write_text(json.dumps(dict(status='running',records=records),indent=2))
        for g in graphs.values():g.reset()
        del graphs,outputs,g,out,obj,packed,scale,x,ids,banks,baseline,direct
        gc.collect();torch.npu.empty_cache()
    (P/f'gemv-tile32-device{selector}.json').write_text(json.dumps(dict(status='passed',records=records,
        max_allocated_bytes=torch.npu.max_memory_allocated(),elapsed_s=time.monotonic()-started),indent=2))

if __name__=='__main__':main()
