import gc,json,statistics,time
from pathlib import Path
import torch,torch_npu
from r102_delta_kernel import update_product

@torch.no_grad()
def main():
    torch.npu.set_device(0);torch.manual_seed(102102)
    sd=torch.randn(1,48,128,128,device='npu:0')*.01
    k=torch.randn(1,48,128,dtype=torch.bfloat16,device='npu:0')*.1
    q=torch.randn_like(k)*.1;v=torch.randn_like(k);beta=torch.rand(1,48,1,device='npu:0').to(torch.bfloat16)
    memory=(sd*k.unsqueeze(-1)).sum(-2)
    state=torch.empty_like(sd);product=torch.empty_like(sd)
    def baseline():
        delta=(v-memory)*beta;s=sd+k.unsqueeze(-1)*delta.unsqueeze(-2)
        state.copy_(s);return (s*q.unsqueeze(-1)).sum(-2)
    def fused():
        update_product(sd[0],k[0],q[0],v[0],memory[0],beta[0],state[0],product[0])
        return product.sum(-2)
    fused();torch.npu.synchronize();results=[]
    for mode,fn in (('torch',baseline),('partial_fusion',fused)):
        stream=torch.npu.Stream();graph=torch.npu.NPUGraph()
        with torch.npu.stream(stream):
            with torch.npu.graph(graph,stream=stream,capture_error_mode='relaxed'):
                for _ in range(36):out=fn()
        torch.npu.synchronize();times=[]
        for _ in range(14):
            begin=time.perf_counter();graph.replay();torch.npu.synchronize();times.append((time.perf_counter()-begin)*1000)
        item={'mode':mode,'repeated_calls':36,'median_ms':statistics.median(times[2:]),'exact':True}
        for seed in range(16):
            torch.manual_seed(seed);sd.copy_(torch.randn_like(sd)*.01);v.copy_(torch.randn_like(v))
            k.copy_(torch.randn_like(k)*.1);q.copy_(torch.randn_like(q)*.1);beta.copy_(torch.rand_like(beta))
            memory.copy_((sd*k.unsqueeze(-1)).sum(-2))
            ref=baseline().clone();ref_state=state.clone()
            graph.replay();torch.npu.synchronize()
            if not torch.equal(out,ref) or not torch.equal(state,ref_state):
                item['exact']=False;item['max_output_abs']=(out-ref).abs().max().item();item['max_state_abs']=(state-ref_state).abs().max().item();break
        print('DELTA_PARTIAL',json.dumps(item),flush=True);results.append(item)
        del graph,out;gc.collect();torch.npu.empty_cache()
    out=Path('/root/ascend-dev/r102');out.mkdir(exist_ok=True)
    (out/'delta-probe.json').write_text(json.dumps(results,indent=2));print('R102_DELTA_COMPLETE',flush=True)

if __name__=='__main__':main()
