"""Compare capture-safe row updates on exact production cache layouts."""
import gc,json,statistics,time
from pathlib import Path
import torch,torch_npu

@torch.no_grad()
def main():
    torch.npu.set_device(0);torch.manual_seed(9696);results=[]
    for name,rows,width,dtype,heads in (
        ('original_kv',262144,512,torch.bfloat16,1),
        ('latent',524288,512,torch.bfloat16,2),
        ('rope',524288,64,torch.bfloat16,2),
        ('raw_index',262144,128,torch.bfloat16,1),
        ('pooled_index',65536,128,torch.float32,1)):
        for T in (1,128,512,8192):
            if name=='pooled_index':T=max(1,(T+6)//4)
            target=torch.zeros(rows,width,dtype=dtype,device='npu:0')
            ids=torch.arange(rows//2,rows//2+T,device='npu:0')
            if heads==2:ids=torch.cat((ids-rows//2,ids))
            updates=torch.randn(len(ids),width,dtype=dtype,device='npu:0')
            expected=target.clone();expected.index_copy_(0,ids,updates)
            for method in ('index_copy','scatter_nd_i64','scatter_nd_i32'):
                try:
                    target.zero_()
                    def write():
                        if method=='index_copy':target.index_copy_(0,ids,updates)
                        else:torch_npu.npu_scatter_nd_update_(target,(ids.int() if method.endswith('i32') else ids).view(-1,1),updates)
                    write();torch.npu.synchronize();assert torch.equal(target,expected)
                    stream=torch.npu.Stream();graph=torch.npu.NPUGraph()
                    with torch.npu.stream(stream):
                        with torch.npu.graph(graph,stream=stream,capture_error_mode='relaxed'):write()
                    torch.npu.synchronize();times=[]
                    for _ in range(14):
                        begin=time.perf_counter();graph.replay();torch.npu.synchronize();times.append((time.perf_counter()-begin)*1000)
                    item={'cache':name,'tokens':T,'method':method,'median_ms':statistics.median(times[2:]),'exact':True}
                    del graph
                except Exception as ex:
                    item={'cache':name,'tokens':T,'method':method,'error':str(ex)[:500]}
                results.append(item);print('WRITE_PROBE',json.dumps(item),flush=True)
            del target,expected,ids,updates;gc.collect();torch.npu.empty_cache()
    Path('/root/ascend-dev/r96').mkdir(exist_ok=True)
    Path('/root/ascend-dev/r96/write-probe.json').write_text(json.dumps(results,indent=2))
    print('R96_PROBE_COMPLETE',flush=True)

if __name__=='__main__':main()
