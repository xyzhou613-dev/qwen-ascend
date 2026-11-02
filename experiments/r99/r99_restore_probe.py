"""Exact inverse-gather alternative to the large MoE output scatter."""
import gc,json,statistics,time
from pathlib import Path
import torch,torch_npu

@torch.no_grad()
def main():
    torch.npu.set_device(0);torch.manual_seed(9999);results=[]
    for T in (128,512,8192):
        N=T*10;H=2560
        ids=torch.randint(0,512,(N,),device='npu:0');order=ids.float().argsort()
        yd=torch.randn(N,H,dtype=torch.bfloat16,device='npu:0')*.1
        w=torch.softmax(torch.randn(T,10,device='npu:0'),-1).to(torch.bfloat16)
        expected=torch.empty_like(yd).index_copy_(0,order,yd)
        expected_y=(expected.view(T,10,H)*w.unsqueeze(-1)).sum(1)
        for method in ('scatter','inverse_gather'):
            def restore():
                if method=='scatter':a=torch.empty_like(yd).index_copy_(0,order,yd)
                else:
                    inverse=torch.empty_like(order).index_copy_(0,order,torch.arange(N,device='npu:0'))
                    a=yd.index_select(0,inverse)
                return a,(a.view(T,10,H)*w.unsqueeze(-1)).sum(1)
            a,y=restore();assert torch.equal(a.view(torch.int16),expected.view(torch.int16))
            assert torch.equal(y.view(torch.int16),expected_y.view(torch.int16))
            del a,y
            stream=torch.npu.Stream();graph=torch.npu.NPUGraph()
            with torch.npu.stream(stream):
                with torch.npu.graph(graph,stream=stream,capture_error_mode='relaxed'):a,y=restore()
            torch.npu.synchronize();times=[]
            for _ in range(14):
                begin=time.perf_counter();graph.replay();torch.npu.synchronize();times.append((time.perf_counter()-begin)*1000)
            item={'tokens':T,'method':method,'median_ms':statistics.median(times[2:]),'restored_exact':True,'weighted_sum_exact':True}
            print('RESTORE_PROBE',json.dumps(item),flush=True);results.append(item)
            # A new expert permutation and new tensor contents must work under
            # graph replay; check arbitrary BF16 bits and weighted output too.
            for seed in (100,101,102):
                torch.manual_seed(seed);order.copy_(torch.randperm(N,device='npu:0'))
                yd.copy_(torch.randint(-32768,32768,yd.shape,dtype=torch.int16,device='npu:0').view(torch.bfloat16))
                graph.replay();torch.npu.synchronize()
                reference=torch.empty_like(yd).index_copy_(0,order,yd)
                reference_y=(reference.view(T,10,H)*w.unsqueeze(-1)).sum(1)
                assert torch.equal(a.view(torch.int16),reference.view(torch.int16)),('bits',T,method,seed)
                assert torch.equal(y.view(torch.int16),reference_y.view(torch.int16)),('weighted',T,method,seed)
                del reference,reference_y
            # Restore identical finite inputs for the next timing variant.
            order.copy_(ids.float().argsort());yd.copy_(expected.index_select(0,order))
            del graph,a,y;gc.collect();torch.npu.empty_cache()
        del yd,w,expected,expected_y,order,ids;gc.collect();torch.npu.empty_cache()
    Path('/root/ascend-dev/r99').mkdir(exist_ok=True)
    Path('/root/ascend-dev/r99/restore-probe.json').write_text(json.dumps(results,indent=2))
    print('R99_RESTORE_ACCEPT',flush=True)

if __name__=='__main__':main()
