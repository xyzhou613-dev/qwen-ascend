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

# A tiny speculative-verification batch touches at most T*TOPK experts.
# Keep a fixed-size subset (including zero-row padding experts) on device,
# avoiding the generic 512-group temporary without host routing reads.
def small_grouped(self,x,group_ends,host_ends=None):
    assert self.expert_major and self._scales is not None and host_ends is None
    counts=group_ends-torch.nn.functional.pad(group_ends[:-1],(1,0))
    active=torch.topk(counts.float(),min(x.shape[0],self.E)).indices
    active=active.float().sort().values.to(torch.int64)
    weight=self.w4.index_select(0,active)
    scales=self._scales.index_select(0,active)
    ends=group_ends.index_select(0,active)
    return torch_npu.npu_grouped_matmul([x],[weight],antiquant_scale=[scales],
        group_list=ends,split_item=3,group_type=0,output_dtype=torch.bfloat16)[0]

E.ExpW4.apply_grouped=small_grouped

@torch.no_grad()
def validate_grouped(obj):
    results=[]
    for rows in (12,20):
        for duplicate in (False,True):
            ids=torch.randperm(obj.E,device='npu')[:rows]
            if duplicate:ids[:rows//2]=7
            ids=ids.float().sort().values.to(torch.int64)
            x=torch.randn(rows,obj.K,device='npu',dtype=torch.bfloat16)*.03
            counts=torch.zeros(obj.E,device='npu',dtype=torch.int64)
            counts.scatter_add_(0,ids,torch.ones_like(ids));ends=counts.cumsum(0)
            actual=obj.apply_grouped(x,ends)
            cart=obj.apply(x,ids).view(rows,rows,obj.N)
            dg=torch.arange(rows,device='npu');ref=cart[dg,dg]
            rel=float((actual.float()-ref.float()).norm()/ref.float().norm().clamp_min(1e-12))
            assert torch.isfinite(actual).all().item() and rel<.025,(obj.N,rows,duplicate,rel)
            results.append(dict(rows=rows,duplicate=duplicate,relative_l2=rel,
                max_abs=float((actual.float()-ref.float()).abs().max())))
    return results

def guard():
    h=json.load(urllib.request.urlopen('http://127.0.0.1:9002/health',timeout=5))
    assert h['service_revision']=='r152' and h['worker_alive']
    assert torch.npu.mem_get_info()[0]>1024**3
    assert time.monotonic()-started<500

@torch.no_grad()
def main():
    records=[]
    for index,ltype in [(24*selector,'linear'),(24*selector+3,'full')]:
        guard(); E._GAMMA_CACHE.clear();os.environ['FNX_MOE_GROUP']='1'
        W=load_layer(f'{QDIR}/L{index:02d}.safetensors',index,'npu:0')
        L=E.FnLayer(index,ltype,W,'npu:0'); del W
        if ltype=='full':
            L.kv=E.KVStore(8192,'npu:0',1)
            if L.index_enabled:
                L.index_k=torch.zeros(8192,128,device='npu',dtype=torch.bfloat16)
                L.index_keys=torch.zeros(2048,128,device='npu',dtype=torch.float32)
        print('LAYER_LOADED',index,torch.npu.memory_allocated(),flush=True)
        validation={'gate_up':validate_grouped(L.exp_gu),'down':validate_grouped(L.exp_dn)}
        print('SMALL_GROUPED_VALIDATED',index,validation,flush=True)
        times={};stream=torch.npu.Stream()
        for T in (1,2,4):
            guard();os.environ['FNX_MOE_GROUP']='0'
            inputs=torch.randn(T,1,E.HCH,device='npu',dtype=torch.bfloat16)*.03
            positions=torch.tensor([4096],device='npu',dtype=torch.int64)
            def fn():
                return L.forward(inputs,positions) if T==1 else L.forward_chunk(inputs,positions,T)
            with torch.npu.stream(stream):
                fn(); stream.synchronize()
                graph=torch.npu.NPUGraph()
                with torch.npu.graph(graph,stream=stream):
                    for _ in range(8):out=fn()
                graph.replay();stream.synchronize()
            print('LAYER_CAPTURED',index,T,torch.npu.memory_allocated(),flush=True)
            times[T]=[]
            for repeat in range(10):
                guard()
                with torch.npu.stream(stream):
                    a=torch.npu.Event(enable_timing=True);b=torch.npu.Event(enable_timing=True)
                    a.record(stream)
                    for _ in range(2):graph.replay()
                    b.record(stream);b.synchronize()
                    times[T].append(a.elapsed_time(b)/16)
                time.sleep(.01)
            graph.reset();del graph,out,inputs,positions,fn
            gc.collect();torch.npu.empty_cache()
        row=dict(layer=index,type=ltype,times_ms=times,median_ms={T:statistics.median(v) for T,v in times.items()},
                 synthetic_hidden=True,context_capacity=8192,position=4096,
                 grouped_validation=validation,
                 note='MoE loops through tokens; batched attention/HC, current forward_chunk, final state only; no draft, state journal, head, IPC or full-model quality.')
        records.append(row)
        result=dict(status='running',selector=selector,records=records,exclusive=False)
        (P/f'verify-layer-loop-device{selector}.json').write_text(json.dumps(result,indent=2))
        print('VERIFY_LAYER_RESULT',index,row['median_ms'],flush=True)
        del L
        E._GAMMA_CACHE.clear();gc.collect();torch.npu.empty_cache()
    result.update(status='passed',elapsed_s=time.monotonic()-started,max_allocated_bytes=torch.npu.max_memory_allocated())
    (P/f'verify-layer-loop-device{selector}.json').write_text(json.dumps(result,indent=2))

if __name__=='__main__':main()
