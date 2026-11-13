"""Real-weight, bounded online W8A8 candidate checks; no Engine."""
import gc,json,os,statistics,sys,time,urllib.request
from pathlib import Path
import torch,torch_npu
from fnx import engine as E
from fnx.load import load_layer,QDIR
P=Path(__file__).resolve().parent;selector=int(sys.argv[1]);started=time.monotonic()
torch.set_num_threads(1);torch.npu.set_device(0)
torch.npu.set_per_process_memory_fraction(1/32,0);torch.manual_seed(15400+selector)
assert str(P/'candidate/fnx/engine.py')==E.__file__

def guard():
    h=json.load(urllib.request.urlopen('http://127.0.0.1:9002/health',timeout=5))
    assert h['service_revision']=='r152' and h['worker_alive']
    assert torch.npu.mem_get_info()[0]>512*1024**2
    assert time.monotonic()-started<650

def compare(a,b):
    a=a.detach().cpu().float().view(-1);b=b.detach().cpu().float().view(-1)
    return dict(relative_l2=float((a-b).norm()/a.norm().clamp_min(1e-12)),
                max_abs=float((a-b).abs().max()),finite=bool(torch.isfinite(b).all()))

def states(L):
    result=[L.conv]
    if getattr(L,'S',None) is not None:result.append(L.S)
    if getattr(L,'ple_conv',None) is not None:result.append(L.ple_conv)
    if L.kv:
        result+=L.kv.k+L.kv.v+[L.index_k,L.index_keys]
    return result

@torch.no_grad()
def main():
    rows=[]
    for index,ltype in [(selector*24,'linear'),(selector*24+3,'full')]:
        guard();E._GAMMA_CACHE.clear();W=load_layer(f'{QDIR}/L{index:02d}.safetensors',index,'npu:0')
        layers={}
        for mode in ('baseline','candidate'):
            os.environ['FNX_DENSE_W8A8']='1' if mode=='candidate' else '0'
            L=E.FnLayer(index,ltype,W,'npu:0')
            if ltype=='full':
                L.kv=E.KVStore(8192,'npu:0',1)
                L.index_k=torch.zeros(8192,128,device='npu',dtype=torch.bfloat16)
                L.index_keys=torch.zeros(2048,128,device='npu',dtype=torch.float32)
            layers[mode]=L
        del W
        torch.npu.synchronize();torch.npu.empty_cache()
        assert layers['candidate'].w8a8_replacements==(['w_qkvz','w_out'] if ltype=='linear' else ['w_q','w_o'])
        print('PAIR_LOADED',index,torch.npu.memory_allocated(),flush=True)
        x=torch.randn(1,1,E.HCH,device='npu',dtype=torch.bfloat16)*.03
        pos=torch.tensor([4096],device='npu',dtype=torch.int64)
        errors=[]
        for i in range(16):
            guard();x.copy_(torch.randn_like(x)*[.003,.03,.3,1.][i%4]);outputs={}
            for mode,L in layers.items():outputs[mode]=L.forward(x,pos).detach().cpu()
            row=compare(outputs['baseline'],outputs['candidate'])
            assert row['finite'],row
            errors.append(row)
        torch.npu.synchronize();torch.npu.empty_cache()
        stream=torch.npu.Stream();graphs={};outs={};times={mode:[] for mode in layers}
        with torch.npu.stream(stream):
            for mode,L in layers.items():
                L.forward(x,pos);stream.synchronize();g=torch.npu.NPUGraph()
                with torch.npu.graph(g,stream=stream,pool=graphs['baseline'].pool() if graphs else None):
                    for _ in range(16):out=L.forward(x,pos)
                graphs[mode]=g;outs[mode]=out;g.replay();stream.synchronize();torch.npu.empty_cache()
            for repeat in range(12):
                guard()
                for mode in (list(layers) if repeat%2==0 else list(reversed(layers))):
                    a=torch.npu.Event(enable_timing=True);b=torch.npu.Event(enable_timing=True);a.record(stream)
                    for _ in range(2):graphs[mode].replay()
                    b.record(stream);b.synchronize();times[mode].append(a.elapsed_time(b)/32)
                time.sleep(.02)
        result=dict(layer=index,type=ltype,decode_errors=errors,times_ms=times,
            median_ms={mode:statistics.median(t) for mode,t in times.items()},
            replacements=layers['candidate'].w8a8_replacements,layout='nd',input='synthetic; 4 amplitudes',
            state_errors={'conv':compare(layers['baseline'].conv,layers['candidate'].conv)})
        if ltype=='linear':result['state_errors']['fp32_state']=compare(layers['baseline'].S,layers['candidate'].S)
        rows.append(result)
        for g in graphs.values():g.reset()
        del layers,L,graphs,g,outs,out,x,pos,outputs
        E._GAMMA_CACHE.clear();gc.collect();torch.npu.empty_cache()
        (P/f'layer-device{selector}.json').write_text(json.dumps(dict(status='running',records=rows),indent=2))
        print('LAYER_RESULT',index,result['median_ms'],flush=True)
    (P/f'layer-device{selector}.json').write_text(json.dumps(dict(status='passed',records=rows,exclusive=False,
        max_allocated_bytes=torch.npu.max_memory_allocated(),elapsed_s=time.monotonic()-started),indent=2))

if __name__=='__main__':main()
