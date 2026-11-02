import gc,json,statistics,time
from pathlib import Path
import torch,torch_npu

@torch.no_grad()
def main():
    torch.npu.set_device(0);torch.manual_seed(105105);records=[]
    for T in (512,4096,8192):
        N=T*10;ids=torch.randint(0,512,(N,),device='npu:0');order=ids.float().argsort()
        arange=torch.arange(N,device='npu:0')
        ref=torch.empty_like(order).index_copy_(0,order,arange)
        for mode in ('index64','scatter64','scatter32','index_float','sort_float','padded8'):
            order.copy_(ids.float().argsort());ref.copy_(torch.empty_like(order).index_copy_(0,order,arange))
            def run():
                a=torch.arange(N,device='npu:0')
                if mode=='index64':return torch.empty_like(order).index_copy_(0,order,a)
                if mode=='scatter64':return torch.empty_like(order).scatter_(0,order,a)
                if mode=='scatter32':return torch.empty_like(order,dtype=torch.int32).scatter_(0,order,a.int())
                if mode=='index_float':return torch.empty_like(order,dtype=torch.float32).index_copy_(0,order,a.float()).long()
                if mode=='sort_float':return order.float().argsort()
                if mode=='padded8':return torch.empty(N,8,dtype=torch.int32,device='npu:0').index_copy_(0,order,a.int()[:,None].expand(-1,8).contiguous())[:,0].contiguous()
            stream=torch.npu.Stream();graph=torch.npu.NPUGraph()
            with torch.npu.stream(stream):
                with torch.npu.graph(graph,stream=stream,capture_error_mode='relaxed'):out=run()
            graph.replay();torch.npu.synchronize();assert torch.equal(out.long(),ref),mode
            samples=[]
            for _ in range(14):
                start=time.perf_counter();graph.replay();torch.npu.synchronize();samples.append((time.perf_counter()-start)*1000)
            exact=True
            for seed in range(4):
                torch.manual_seed(seed);order.copy_(torch.randperm(N,device='npu:0'))
                graph.replay();torch.npu.synchronize();ref.copy_(torch.empty_like(order).index_copy_(0,order,arange))
                exact &=torch.equal(out.long(),ref)
            item={'tokens':T,'mode':mode,'median_ms':statistics.median(samples[2:]),'exact':exact};records.append(item);print('INVERSE_PROBE',json.dumps(item),flush=True)
            del graph,out;gc.collect();torch.npu.empty_cache()
    d=Path('/root/ascend-dev/r105');d.mkdir(exist_ok=True);(d/'inverse-probe.json').write_text(json.dumps(records,indent=2))
    print('R105_INVERSE_COMPLETE',flush=True)

if __name__=='__main__':main()
