import gc,json,os,statistics,time
from pathlib import Path
import torch,torch_npu
from r103_kernels import combine8192,conv_step

def timing(fn,count):
    stream=torch.npu.Stream();graph=torch.npu.NPUGraph()
    with torch.npu.stream(stream):
        with torch.npu.graph(graph,stream=stream,capture_error_mode='relaxed'):
            for _ in range(count):out=fn()
    torch.npu.synchronize();samples=[]
    for _ in range(14):
        start=time.perf_counter();graph.replay();torch.npu.synchronize();samples.append((time.perf_counter()-start)*1000)
    return graph,out,statistics.median(samples[2:])

@torch.no_grad()
def main():
    torch.npu.set_device(0);torch.manual_seed(103103);records=[]
    if os.environ.get('PROBE_PART','moe')=='moe':
        T=8192;N=T*10;H=2560
        yd=torch.randn(N,H,dtype=torch.bfloat16,device='npu:0')*.1
        inv=torch.randperm(N,device='npu:0').int().view(T,10)
        weights=torch.softmax(torch.randn(T,10,device='npu:0'),-1).to(torch.bfloat16).unsqueeze(-1)
        output=torch.empty(T,H,dtype=yd.dtype,device='npu:0')
        def base():return (yd.index_select(0,inv.reshape(-1).long()).view(T,10,H)*weights).sum(1)
        def fused():combine8192(yd,inv,weights,output);return output
        fused();torch.npu.synchronize()
        for name,fn in (('torch',base),('pto',fused)):
            graph,out,ms=timing(fn,1);exact=True;worst=0
            for seed in range(4):
                torch.manual_seed(seed);yd.copy_(torch.randn_like(yd)*.1);inv.copy_(torch.randperm(N,device='npu:0').int().view(T,10))
                ref=base();graph.replay();torch.npu.synchronize();worst=max(worst,(out.float()-ref.float()).abs().max().item());exact &=torch.equal(out,ref)
            item={'part':'moe','mode':name,'calls':1,'median_ms':ms,'exact':exact,'max_abs':worst};print('PTO_PROBE',json.dumps(item),flush=True);records.append(item)
            del graph,out;gc.collect();torch.npu.empty_cache()
    else:
        history=torch.randn(10240,3,dtype=torch.bfloat16,device='npu:0');current=torch.randn(10240,1,dtype=history.dtype,device=history.device)
        weights=torch.randn(10240,4,dtype=history.dtype,device=history.device);new=torch.empty_like(history);output=torch.empty(10240,dtype=history.dtype,device=history.device)
        def base():
            buf=torch.cat((history,current),-1);y=(buf*weights).sum(-1);new.copy_(buf[:,1:]);return y
        def fused():conv_step(history,current,weights,new,output);return output
        fused();torch.npu.synchronize()
        for name,fn in (('torch',base),('pto',fused)):
            graph,out,ms=timing(fn,36);exact=True;worst=0
            for seed in range(32):
                torch.manual_seed(seed);history.copy_(torch.randn_like(history));current.copy_(torch.randn_like(current));weights.copy_(torch.randn_like(weights))
                ref=base().clone();ref_new=new.clone();graph.replay();torch.npu.synchronize()
                worst=max(worst,(out.float()-ref.float()).abs().max().item());exact &=torch.equal(out,ref) and torch.equal(new,ref_new)
            item={'part':'conv','mode':name,'calls':36,'median_ms':ms,'exact':exact,'max_abs':worst};print('PTO_PROBE',json.dumps(item),flush=True);records.append(item)
            del graph,out;gc.collect();torch.npu.empty_cache()
    out=Path('/root/ascend-dev/r103');out.mkdir(exist_ok=True)
    (out/(os.environ.get('PROBE_PART','moe')+'-probe.json')).write_text(json.dumps(records,indent=2));print('R103_PROBE_COMPLETE',flush=True)

if __name__=='__main__':main()
