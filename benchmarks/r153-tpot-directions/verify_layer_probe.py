"""Bounded real-layer performance exploration, synthetic hidden states.

Not an MTP verifier or a model-quality test. No second Engine is loaded.
"""
import gc,json,os,statistics,sys,time,urllib.request
from pathlib import Path
os.environ['FNX_CTX']='8192'
import torch,torch_npu
from fnx import engine as E
from fnx.load import load_layer,QDIR
P=Path(__file__).resolve().parent
selector=int(sys.argv[1]); started=time.monotonic()
torch.set_num_threads(1);torch.npu.set_device(0)
torch.npu.set_per_process_memory_fraction(1/32,0)
torch.manual_seed(15342+selector)

def guard():
    h=json.load(urllib.request.urlopen('http://127.0.0.1:9002/health',timeout=5))
    assert h['service_revision']=='r152' and h['worker_alive']
    assert torch.npu.mem_get_info()[0]>1024**3
    assert time.monotonic()-started<500

@torch.no_grad()
def main():
    records=[]
    for index,ltype in [(24*selector,'linear'),(24*selector+3,'full')]:
        guard(); E._GAMMA_CACHE.clear()
        W=load_layer(f'{QDIR}/L{index:02d}.safetensors',index,'npu:0')
        L=E.FnLayer(index,ltype,W,'npu:0'); del W
        if ltype=='full':
            L.kv=E.KVStore(8192,'npu:0',1)
            if L.index_enabled:
                L.index_k=torch.zeros(8192,128,device='npu',dtype=torch.bfloat16)
                L.index_keys=torch.zeros(2048,128,device='npu',dtype=torch.float32)
        print('LAYER_LOADED',index,torch.npu.memory_allocated(),flush=True)
        graphs={};outputs={};inputs={};positions={};stream=torch.npu.Stream()
        for T in (1,2,4):
            guard()
            inputs[T]=torch.randn(T,1,E.HCH,device='npu',dtype=torch.bfloat16)*.03
            positions[T]=torch.tensor([4096],device='npu',dtype=torch.int64)
            def fn():
                return L.forward(inputs[T],positions[T]) if T==1 else L.forward_chunk(inputs[T],positions[T],T)
            with torch.npu.stream(stream):
                fn(); stream.synchronize()
                graph=torch.npu.NPUGraph()
                with torch.npu.graph(graph,stream=stream):
                    for _ in range(8):out=fn()
                outputs[T]=out;graphs[T]=graph
                graph.replay();stream.synchronize()
            print('LAYER_CAPTURED',index,T,torch.npu.memory_allocated(),flush=True)
        times={T:[] for T in graphs}
        for repeat in range(10):
            guard()
            for T in ([1,2,4] if repeat%2==0 else [4,2,1]):
                with torch.npu.stream(stream):
                    a=torch.npu.Event(enable_timing=True);b=torch.npu.Event(enable_timing=True)
                    a.record(stream)
                    for _ in range(2):graphs[T].replay()
                    b.record(stream);b.synchronize()
                    times[T].append(a.elapsed_time(b)/16)
                time.sleep(.01)
        row=dict(layer=index,type=ltype,times_ms=times,median_ms={T:statistics.median(v) for T,v in times.items()},
                 synthetic_hidden=True,context_capacity=8192,position=4096,
                 note='Current forward_chunk, final state only; no draft, state journal, head, IPC or full-model quality.')
        records.append(row)
        result=dict(status='running',selector=selector,records=records,exclusive=False)
        (P/f'verify-layer-device{selector}.json').write_text(json.dumps(result,indent=2))
        print('VERIFY_LAYER_RESULT',index,row['median_ms'],flush=True)
        for graph in graphs.values():graph.reset()
        del graphs,graph,L,inputs,positions,outputs,out,fn
        E._GAMMA_CACHE.clear();gc.collect();torch.npu.empty_cache()
    result.update(status='passed',elapsed_s=time.monotonic()-started,max_allocated_bytes=torch.npu.max_memory_allocated())
    (P/f'verify-layer-device{selector}.json').write_text(json.dumps(result,indent=2))

if __name__=='__main__':main()
