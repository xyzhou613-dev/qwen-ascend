"""Native routing fusion with explicit BF16 product rounding when needed."""
import gc,json,statistics,time
from pathlib import Path
import torch,torch_npu

@torch.no_grad()
def main():
    torch.npu.set_device(0);torch.manual_seed(101101);records=[]
    for T in (8,128,512,8192):
        H=2560;K=10;N=T*K
        yd=torch.randn(N,H,dtype=torch.bfloat16,device='npu:0')*.1
        ids=torch.randint(0,512,(N,),device='npu:0');order=ids.float().argsort()
        w=torch.softmax(torch.randn(T,K,device='npu:0'),-1).to(torch.bfloat16)
        ones=torch.ones_like(w)
        inverse=torch.empty_like(order).index_copy_(0,order,torch.arange(N,device='npu:0'))
        expected=(yd.index_select(0,inverse).view(T,K,H)*w.unsqueeze(-1)).sum(1)
        for mode in ('inverse','native_inverse','rounded_inverse'):
            torch.manual_seed(101101+T)
            yd.copy_(torch.randn_like(yd)*.1);order.copy_(ids.float().argsort())
            w.copy_(torch.softmax(torch.randn(T,K,device='npu:0'),-1).to(torch.bfloat16))
            inverse.copy_(torch.empty_like(order).index_copy_(0,order,torch.arange(N,device='npu:0')))
            expected.copy_((yd.index_select(0,inverse).view(T,K,H)*w.unsqueeze(-1)).sum(1))
            def run():
                if mode=='inverse':
                    inv=torch.empty_like(order).index_copy_(0,order,torch.arange(N,device='npu:0'))
                    return (yd.index_select(0,inv).view(T,K,H)*w.unsqueeze(-1)).sum(1)
                index=torch.empty_like(order).index_copy_(0,order,torch.arange(N,device='npu:0')).int()
                if mode.startswith('rounded'):
                    weighted=yd*w.reshape(-1).index_select(0,order)[:,None]
                    return torch_npu.npu_moe_token_unpermute(weighted,index,ones)
                return torch_npu.npu_moe_token_unpermute(yd,index,w)
            try:
                output=run();torch.npu.synchronize()
                difference=(output.float()-expected.float()).abs()
                item={'tokens':T,'mode':mode,'max_abs':difference.max().item(),'equal_fraction':(output==expected).float().mean().item()}
                if item['max_abs']>.01:
                    item['mapping_rejected']=True;records.append(item);print('MOE_FUSION',json.dumps(item),flush=True);continue
                stream=torch.npu.Stream();graph=torch.npu.NPUGraph()
                with torch.npu.stream(stream):
                    with torch.npu.graph(graph,stream=stream,capture_error_mode='relaxed'):output=run()
                torch.npu.synchronize();times=[]
                for _ in range(14):
                    begin=time.perf_counter();graph.replay();torch.npu.synchronize();times.append((time.perf_counter()-begin)*1000)
                item['median_ms']=statistics.median(times[2:]);item['dynamic_exact']=True
                for seed in (1,2,3):
                    torch.manual_seed(seed);order.copy_(torch.randperm(N,device='npu:0'));yd.copy_(torch.randn_like(yd)*.1)
                    w.copy_(torch.softmax(torch.randn(T,K,device='npu:0'),-1).to(torch.bfloat16))
                    graph.replay();torch.npu.synchronize()
                    inv=torch.empty_like(order).index_copy_(0,order,torch.arange(N,device='npu:0'))
                    reference=(yd.index_select(0,inv).view(T,K,H)*w.unsqueeze(-1)).sum(1)
                    if not torch.equal(output,reference):item['dynamic_exact']=False
                del graph,output;gc.collect();torch.npu.empty_cache()
            except Exception as ex:item={'tokens':T,'mode':mode,'error':str(ex)[:500]}
            records.append(item);print('MOE_FUSION',json.dumps(item),flush=True)
        del yd,w,ones,order,ids,inverse,expected;gc.collect();torch.npu.empty_cache()
    out=Path('/root/ascend-dev/r101');out.mkdir(exist_ok=True)
    (out/'moe-probe.json').write_text(json.dumps(records,indent=2));print('R101_MOE_COMPLETE',flush=True)

if __name__=='__main__':main()
