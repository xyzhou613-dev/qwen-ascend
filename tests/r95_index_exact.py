"""Real-weight, state-and-selected-ID equivalence plus captured timings."""
import json,os,statistics,time,gc
from pathlib import Path
import torch,torch_npu
from fnx import engine as E
from fnx.load import load_layer,QDIR

def mode(unique,sort):os.environ.update(FNX_INDEX_UNIQUE=str(unique),FNX_INDEX_BLOCK_SORT=str(sort))

@torch.no_grad()
def main():
    torch.npu.set_device(0);torch.manual_seed(93294)
    W=load_layer(os.path.join(QDIR,'L03.safetensors'),3,'npu:0')
    L=E.FnLayer(3,'full',W,'npu:0');del W
    L.kv=E.KVStore(262144,'npu:0',1)
    L.index_k=torch.randn(262144,128,dtype=torch.bfloat16,device='npu:0')*.03
    L.index_keys=torch.randn(262144//4,128,dtype=torch.float32,device='npu:0')*.03
    cases=[]
    for T in (1,2,3,4,7,8,64,128,512,8192):
        for start in (0,1,2,3,2045,2046,2047,2048,16381,16382,16383,16384,196605,262144-T):
            if start+T<=262144:cases.append((start,T))
    for number,(start,T) in enumerate(cases):
        h=torch.randn(T,1,E.H,dtype=torch.bfloat16,device='npu:0')*.03
        pos=torch.arange(start,start+T,device='npu:0')
        L.index_scan_cap=min(262144,max(16384,(start+T+3)//4*4))
        raw=L.index_k.clone();keys=L.index_keys.clone()
        mode(0,0);expected,expected_valid=L._indexer_indices(h,pos)
        after_raw=L.index_k.clone();after_keys=L.index_keys.clone()
        for unique,sort in ((1,0),):
            L.index_k.copy_(raw);L.index_keys.copy_(keys);mode(unique,sort)
            actual,valid=L._indexer_indices(h,pos)
            assert torch.equal(actual,expected) and torch.equal(valid,expected_valid),('selected',start,T,unique,sort)
            assert torch.equal(L.index_k,after_raw) and torch.equal(L.index_keys,after_keys),('cache',start,T,unique,sort)
        if number%14==13:print('INDEX_BOUNDARY_EXACT',T,flush=True)
    results=[]
    for T in (1,128,512,8192):
        pos=torch.arange(196611,196611+T,device='npu:0')
        h=torch.randn(T,1,E.H,dtype=torch.bfloat16,device='npu:0')*.03
        L.index_scan_cap=262144
        for unique,sort in ((0,0),(1,0)):
            pos.copy_(torch.arange(196611,196611+T,device='npu:0'))
            mode(unique,sort);stream=torch.npu.Stream()
            with torch.npu.stream(stream):
                L._indexer_indices(h,pos);torch.npu.synchronize()
                graph=torch.npu.NPUGraph()
                with torch.npu.graph(graph,stream=stream,capture_error_mode='relaxed'):
                    result=L._indexer_indices(h,pos)
            torch.npu.synchronize();times=[]
            for _ in range(12):
                begin=time.perf_counter();graph.replay();torch.npu.synchronize();times.append((time.perf_counter()-begin)*1000)
            item={'tokens':T,'unique':unique,'block_sort':sort,'median_ms':statistics.median(times[2:])}
            results.append(item);print('INDEX_PERF',json.dumps(item),flush=True)
            # Captured positions remain dynamic, including every mod-4 class.
            for begin in (2045,2046,2047,2048,262144-T):
                pos.copy_(torch.arange(begin,begin+T,device='npu:0'))
                graph.replay();torch.npu.synchronize()
                saved=(result[0].clone(),result[1].clone())
                mode(0,0);reference=L._indexer_indices(h,pos)
                assert all(torch.equal(a,b) for a,b in zip(saved,reference)),('graph',begin,T,unique,sort)
            del graph,result;gc.collect();torch.npu.empty_cache()
    Path(os.environ.get('FNX_INDEX_TEST_OUTPUT','/tmp/r95-index-test.json')).write_text(json.dumps(results,indent=2))
    print('R94_INDEX_ACCEPT',len(cases),flush=True)

if __name__=='__main__':main()
