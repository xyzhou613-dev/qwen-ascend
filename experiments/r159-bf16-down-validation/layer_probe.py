"""Three-way real-layer comparison; immutable weights, separate recurrent state."""
import gc,json,os,statistics,sys,time
from pathlib import Path
import torch,torch_npu
from fnx import engine as E
from fnx.load import load_layer,QDIR
from fnx import kmajor_down as D

P=Path(__file__).resolve().parent; device=int(sys.argv[1]);started=time.monotonic()
torch.set_num_threads(1);torch.npu.set_device(0)
torch.npu.set_per_process_memory_fraction(1/16,0);torch.manual_seed(15950+device)
rounded_apply=D.apply
def mode_apply(projection,x,ids):
    if os.environ.get('BENCH_DOWN_MODE')!='old':return rounded_apply(projection,x,ids)
    # Initialize both handles once using the validated rounded wrapper.
    if D._lib is None:rounded_apply(projection,x,ids)
    out=torch.empty((10,2560),device=x.device,dtype=torch.bfloat16)
    assert D._lib.fnx_w4_kmajor(projection.w4.data_ptr(),projection._scales.data_ptr(),ids.data_ptr(),x.data_ptr(),out.data_ptr(),256,torch.npu.current_stream(x.device).npu_stream)==0
    return out
D.apply=mode_apply
def guard():
    assert (P/'lease.active').exists() and time.monotonic()-started<550
    assert time.time()<float((P/'window-start.txt').read_text())+2500
def error(a,b):
    a=a.float();b=b.float()
    return dict(relative_l2=float((a-b).norm()/a.norm().clamp_min(1e-12)),max_abs=float((a-b).abs().max()),finite=bool(torch.isfinite(b).all()))
def mode(name):
    os.environ['FNX_KMAJOR_DOWN']='0' if name=='w8a8' else '1'
    os.environ['BENCH_DOWN_MODE']=name

@torch.no_grad()
def main():
    records=[]
    for index,kind in ((device*24,'linear'),(device*24+3,'full')):
        guard();E._GAMMA_CACHE.clear()
        weights=load_layer(f'{QDIR}/L{index:02d}.safetensors',index,'npu:0')
        layers={}
        for name in ('w8a8','old','rounded'):
            layer=E.FnLayer(index,kind,weights,'npu:0')
            if kind=='full':
                layer.kv=E.KVStore(8192,'npu:0',1)
                layer.index_k=torch.zeros(8192,128,device='npu',dtype=torch.bfloat16)
                layer.index_keys=torch.zeros(2048,128,device='npu',dtype=torch.float32)
            layers[name]=layer
        del weights;torch.npu.empty_cache()
        x=torch.empty(1,1,E.HCH,device='npu',dtype=torch.bfloat16)
        pos=torch.tensor([4096],device='npu',dtype=torch.int64)
        errors=[];hit_start=D.hits
        for sample in range(16):
            guard();x.copy_(torch.randn_like(x)*(.003,.03,.3,1.)[sample%4]);outputs={}
            for name,layer in layers.items():
                mode(name);outputs[name]=layer.forward(x,pos).detach().cpu()
            row={name:error(outputs['w8a8'],outputs[name]) for name in ('old','rounded')}
            assert row['rounded']['finite'] and row['rounded']['relative_l2']<.001,row
            errors.append(row)
        state={}
        ref=layers['w8a8']
        for name,layer in layers.items():
            state[name+'/conv']=torch.equal(ref.conv,layer.conv)
            if kind=='linear':state[name+'/FP32_S']=torch.equal(ref.S,layer.S)
            else:
                state[name+'/kv']=all(torch.equal(a,b) for a,b in zip(ref.kv.k+ref.kv.v,layer.kv.k+layer.kv.v))
                state[name+'/index_k']=torch.equal(ref.index_k,layer.index_k)
                state[name+'/index_keys']=torch.equal(ref.index_keys,layer.index_keys)
        assert all(state.values()),state
        assert D.hits-hit_start==32,'direct decode path was bypassed'
        stream=torch.npu.Stream();graphs={};times={name:[] for name in layers}
        torch.npu.empty_cache()
        with torch.npu.stream(stream):
            for name,layer in layers.items():
                mode(name);layer.forward(x,pos);stream.synchronize();g=torch.npu.NPUGraph()
                with torch.npu.graph(g,stream=stream,pool=next(iter(graphs.values())).pool() if graphs else None):
                    for _ in range(16):out=layer.forward(x,pos)
                graphs[name]=g;g.replay();stream.synchronize()
            for rep in range(16):
                guard()
                for name in (tuple(graphs) if rep%2==0 else tuple(reversed(graphs))):
                    begin=torch.npu.Event(enable_timing=True);end=torch.npu.Event(enable_timing=True)
                    begin.record(stream)
                    for _ in range(2):graphs[name].replay()
                    end.record(stream);end.synchronize()
                    times[name].append(begin.elapsed_time(end)/32)
            for g in graphs.values():g.reset()
        row=dict(layer=index,kind=kind,median_ms={k:statistics.median(v) for k,v in times.items()},
            samples_ms=times,output_errors=errors,state_checks_before_timing=state,
            exact_dispatch_hits_in_numeric_check=32)
        records.append(row)
        print('REAL_LAYER_RESULT',index,row['median_ms'],flush=True)
        (P/f'layers-device{device}.json').write_text(json.dumps(dict(status='running',records=records),indent=2)+'\n')
        del layers,layer,ref,graphs,g,out,x,pos,outputs
        E._GAMMA_CACHE.clear();gc.collect();torch.npu.empty_cache()
    result=dict(status='passed_component',records=records,elapsed_s=time.monotonic()-started,
        peak_allocated_bytes=torch.npu.max_memory_allocated(),
        scope='4 real layers; weighted proxy is not HTTP TPOT or full quality acceptance')
    (P/f'layers-device{device}.json').write_text(json.dumps(result,indent=2)+'\n')

if __name__=='__main__':main()
