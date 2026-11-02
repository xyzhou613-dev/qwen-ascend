"""Measure the actual rank-3 position-major KV update and a flat row view."""
import gc,json,statistics,time
from pathlib import Path
import torch,torch_npu

@torch.no_grad()
def main():
    torch.npu.set_device(0);results=[]
    P=262144;target=torch.zeros(P,2,256,dtype=torch.bfloat16,device='npu:0')
    for T in (1,128,512,8192):
        target.zero_()
        pos=torch.arange(196611,196611+T,device='npu:0')
        source=torch.randint(-32768,32767,(T,2,256),dtype=torch.int16,device='npu:0').view(torch.bfloat16)
        expected=target.clone();expected.index_copy_(0,pos,source)
        for method in ('rank3','flat2'):
            pos.copy_(torch.arange(196611,196611+T,device='npu:0'))
            expected.zero_();expected.index_copy_(0,pos,source)
            target.zero_()
            def write():
                if method=='rank3':target.index_copy_(0,pos,source)
                else:target.view(P,512).index_copy_(0,pos,source.view(T,512))
            write();torch.npu.synchronize();assert torch.equal(target.view(torch.int16),expected.view(torch.int16))
            stream=torch.npu.Stream();graph=torch.npu.NPUGraph();repeats=16
            with torch.npu.stream(stream):
                with torch.npu.graph(graph,stream=stream,capture_error_mode='relaxed'):
                    for _ in range(repeats):write()
            torch.npu.synchronize();times=[]
            for _ in range(12):
                begin=time.perf_counter();graph.replay();torch.npu.synchronize();times.append((time.perf_counter()-begin)*1000/repeats)
            item={'tokens':T,'method':method,'median_per_write_ms':statistics.median(times[2:]),'exact_bf16_bits':True}
            print('RANK_PROBE',json.dumps(item),flush=True);results.append(item)
            # Changing device-side positions must remain capture-safe.
            for start in (0,1,2047,P-T):
                pos.copy_(torch.arange(start,start+T,device='npu:0'));target.zero_();expected.zero_()
                graph.replay();expected.index_copy_(0,pos,source)
                assert torch.equal(target.view(torch.int16),expected.view(torch.int16))
            del graph;gc.collect();torch.npu.empty_cache()
        del expected
    Path('/root/ascend-dev/r97').mkdir(exist_ok=True)
    Path('/root/ascend-dev/r97/rank-probe.json').write_text(json.dumps(results,indent=2))
    print('R97_RANK_ACCEPT',flush=True)

if __name__=='__main__':main()
