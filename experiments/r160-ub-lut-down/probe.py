"""Single-weight LUT correctness; optional exclusive NPU-event timing."""
import argparse,ctypes,hashlib,json,os,statistics,time,urllib.request
from pathlib import Path
import torch,torch_npu
from safetensors import safe_open
from fnx import engine as E
from fnx.moe_epilogue import merge as production_merge
P=Path(__file__).resolve().parent
a=argparse.ArgumentParser();a.add_argument('device',type=int);a.add_argument('--timing',action='store_true');args=a.parse_args()
started=time.monotonic();observed=None
def guard():
    global observed
    assert time.monotonic()-started<160
    if args.timing:
        assert (P/'lease.active').exists()
        assert time.time()<json.loads((P/'window.json').read_text())['restore_trigger_epoch']
        for p in Path('/proc').glob('[0-9]*'):
            try:c=p.joinpath('cmdline').read_bytes().split(b'\0')
            except (FileNotFoundError,PermissionError,ProcessLookupError):continue
            assert b'fnx.server' not in c and b'fnx.pipeline_worker' not in c,'REFUSED_SHARED_TIMING'
    else:
        h=json.load(urllib.request.urlopen('http://127.0.0.1:9002/health',timeout=5))
        assert h['service_revision']=='r152' and h['worker_alive']
        if observed is None:observed=h['pipeline_transfers']
        assert h['pipeline_transfers']==observed,'ONLINE_REQUEST_STARTED_STOP_PROBE'

def lib(path,symbol):
    obj=ctypes.CDLL(str(path));f=getattr(obj,symbol)
    f.argtypes=[ctypes.c_void_p]*5+[ctypes.c_int,ctypes.c_void_p];f.restype=ctypes.c_int
    return obj,f

def error(a,b):
    a=a.cpu().float();b=b.cpu().float()
    return dict(relative_l2=float((a-b).norm()/a.norm().clamp_min(1e-12)),max_abs=float((a-b).abs().max()),
        exact_fraction=float((a==b).float().mean()),finite=bool(torch.isfinite(b).all()))

@torch.no_grad()
def main():
    guard();torch.set_num_threads(1);torch.manual_seed(16000+args.device)
    torch.npu.set_device(0);torch.npu.set_per_process_memory_fraction(1/32,0)
    cfg=json.loads(Path('/models/fnx-diagnostics/r155-smooth-tpot-ab-20261113/production-selected-config.json').read_text())
    layer=24*args.device
    with safe_open(cfg['FNX_W4A16_DIR']+f'-em/L{layer:02d}.w4.safetensors',framework='pt',device='cpu') as f:
        k='mlp.experts.down_proj'
        obj=E.ExpW4(f.get_tensor(k+'.w4').npu(),f.get_tensor(k+'.s8t').npu(),f.get_tensor(k+'.base').npu(),64,expert_major=True)
    oldlib,old=lib(Path('/models/fnx-diagnostics/r156-moe-kmajor-20261113/libfnx_w4_kmajor.so'),'fnx_w4_kmajor')
    reflib,ref=lib(Path('/models/fnx-diagnostics/r158-bf16-down-20261113/libfnx_w4_bf16w.so'),'fnx_w4_bf16w')
    lutlib,lut=lib(P/'libfnx_w4_lut.so','fnx_w4_bf16w')
    ep=oldlib.fnx_diag_epilogue;ep.argtypes=[ctypes.c_void_p]*6;ep.restype=ctypes.c_int
    x=torch.empty((10,640),device='npu',dtype=torch.bfloat16)
    ids=torch.arange(10,device='npu',dtype=torch.int64)
    weights=torch.empty((1,10),device='npu',dtype=torch.bfloat16)
    shared=torch.empty((1,2560),device='npu',dtype=torch.bfloat16)
    gate=torch.empty((1,1),device='npu',dtype=torch.bfloat16)
    labels=('production','old','rounded','lut128','lut256')
    def diagonal(label,ix):
        fn=old if label=='old' else ref if label=='rounded' else lut
        tile=128 if label=='lut128' else 256
        out=torch.empty((10,2560),device='npu',dtype=torch.bfloat16)
        assert fn(obj.w4.data_ptr(),obj._scales.data_ptr(),ix.data_ptr(),x.data_ptr(),out.data_ptr(),tile,torch.npu.current_stream().npu_stream)==0
        return out
    def run(label,ix):
        if label=='production':return production_merge(obj.apply(x,ix),weights,shared,gate)
        diag=diagonal(label,ix);out=torch.empty_like(shared)
        assert ep(diag.data_ptr(),weights.data_ptr(),shared.data_ptr(),gate.data_ptr(),out.data_ptr(),torch.npu.current_stream().npu_stream)==0
        return out
    def fresh(sample):
        ix=(torch.arange(10)+sample*10)%512
        if sample==52:ix.fill_(511)
        ids.copy_(ix.npu());amp=(.003,.03,.3,1.)[sample%4]
        x.copy_(torch.randn_like(x)*amp);shared.copy_(torch.randn_like(shared)*amp)
        weights.copy_(torch.softmax(torch.randn(1,10,device='npu'),-1).bfloat16())
        gate.fill_((-15,-2,0,2,15)[sample%5]);return ix.tolist()
    records=[];covered=set()
    for sample in range(53):
        guard();covered.update(fresh(sample));expected=diagonal('rounded',ids)
        expected_merge=run('rounded',ids);production=run('production',ids);row=dict(sample=sample,lut={})
        for label in ('lut128','lut256'):
            got=diagonal(label,ids);merged=run(label,ids)
            assert torch.equal(expected.cpu(),got.cpu()),('DIAGONAL_DIFFERS_FROM_R158',label,error(expected,got))
            assert torch.equal(expected_merge.cpu(),merged.cpu()),('MERGE_DIFFERS_FROM_R158',label)
            row['lut'][label]=error(production,merged);assert row['lut'][label]['finite']
        records.append(row)
    assert len(covered)==512
    stream=torch.npu.Stream();fresh_graph={}
    with torch.npu.stream(stream):
        for label in ('lut128','lut256'):
            run(label,ids);stream.synchronize();g=torch.npu.NPUGraph()
            with torch.npu.graph(g,stream=stream):captured=run(label,ids)
            stream.synchronize()
            for sample in range(8):
                guard();fresh(sample+60);eager=run(label,ids);g.replay();stream.synchronize()
                assert torch.equal(eager.cpu(),captured.cpu()),('STALE_GRAPH',label)
                assert torch.equal(eager.cpu(),run('rounded',ids).cpu()),('FRESH_R158_DIFF',label)
            fresh_graph[label]=True;g.reset()
    result=dict(status='numeric_pass',device=args.device,layer=layer,experts_covered=len(covered),samples=53,
        bitwise_equal_to_r158=True,fresh_graph_verified=fresh_graph,records=records,
        exclusive_timing=args.timing,peak_allocated_bytes=torch.npu.max_memory_allocated(),
        elapsed_s=time.monotonic()-started,code_sha256={n:hashlib.sha256((P/n).read_bytes()).hexdigest()
            for n in ('lut_down.cpp','libfnx_w4_lut.so','probe.py')})
    if args.timing:
        banks=torch.stack([(torch.arange(10)+i*10)%512 for i in range(64)]).npu()
        graphs={};times={label:[] for label in labels}
        torch.npu.empty_cache()
        with torch.npu.stream(stream):
            for label in labels:
                run(label,banks[0]);stream.synchronize();g=torch.npu.NPUGraph()
                with torch.npu.graph(g,stream=stream,pool=next(iter(graphs.values())).pool() if graphs else None):
                    for bank in banks:out=run(label,bank)
                graphs[label]=g;g.replay();stream.synchronize()
            for rep in range(16):
                guard()
                for label in (labels if rep%2==0 else tuple(reversed(labels))):
                    begin=torch.npu.Event(enable_timing=True);end=torch.npu.Event(enable_timing=True);begin.record(stream)
                    for _ in range(2):graphs[label].replay()
                    end.record(stream);end.synchronize();times[label].append(begin.elapsed_time(end)*1000/128)
            for g in graphs.values():g.reset()
        result.update(status='exclusive_component_complete',median_us={k:statistics.median(v) for k,v in times.items()},
            samples_us=times,peak_allocated_bytes=torch.npu.max_memory_allocated(),elapsed_s=time.monotonic()-started)
    mode='timing' if args.timing else 'numeric'
    (P/f'{mode}-device{args.device}.json').write_text(json.dumps(result,indent=2)+'\n')
    print('LUT_RESULT',mode,args.device,{k:v for k,v in result.items() if k not in ('records','samples_us','code_sha256')},flush=True)

if __name__=='__main__':main()
