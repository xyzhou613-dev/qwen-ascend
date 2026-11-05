"""Gate BF16 product rounding and grouped FP32 accumulation."""
import gc
import os
import json
import statistics
import time
from pathlib import Path
import torch
import torch_npu
from fnx.weighted_reduce import reduce


def same(a,b):
    return torch.equal(a.view(torch.int16),b.view(torch.int16))


def benchmark(fn,repetitions):
    graph=torch.npu.NPUGraph()
    torch.npu.synchronize()
    with torch.npu.graph(graph):
        for _ in range(repetitions):out=fn()
    graph.replay();torch.npu.synchronize()
    samples=[]
    for _ in range(10):
        start=time.perf_counter();graph.replay();torch.npu.synchronize()
        samples.append((time.perf_counter()-start)*1000/repetitions)
    return statistics.median(samples[2:])


@torch.no_grad()
def main():
    device="npu:0";H=2560
    torch.manual_seed(117)
    records=[]
    orders={}
    for kind in ("mix","moe"):
        R=4 if kind=="mix" else 10
        values=torch.randn(1,R,H,device=device,dtype=torch.bfloat16)
        weights=torch.ones_like(values) if kind=="mix" else torch.ones(1,R,device=device,dtype=torch.bfloat16)
        values[0,:,0]=0;values[0,0,0]=1e8;values[0,1,0]=1;values[0,2,0]=-1e8;values[0,3,0]=1
        values[0,:,1]=0;values[0,0,1]=1e8;values[0,1,1]=-1e8;values[0,2,1]=1;values[0,3,1]=1
        reference=lambda:(values*weights if kind=="mix" else values*weights.unsqueeze(-1)).mean(1) if kind=="mix" else (values*weights.unsqueeze(-1)).sum(1)
        expected=reference()
        accepted=[]
        for order in (0,):
            actual=reduce(values,weights,mean=kind=="mix",order=order)
            exact=same(actual,expected)
            print("REDUCE_ORDER",kind,order,exact,(actual.float()-expected.float()).abs().max().item(),flush=True)
            if exact:accepted.append(order)
        assert accepted,kind
        orders[kind]=accepted[0]
    for kind in ("mix","moe"):
        for B in (1,128,512,8192):
            R=4 if kind=="mix" else 10
            values=torch.randn(B,R,H,device=device,dtype=torch.bfloat16)*.1
            weights=torch.rand_like(values) if kind=="mix" else torch.rand(B,R,device=device,dtype=torch.bfloat16)
            inverse=torch.randperm(B*R,device=device) if kind=="moe" and B>1 else None
            source=values.view(-1,H)
            def reference():
                if kind=="mix":return (values*weights).mean(1)
                restored=source.index_select(0,inverse).view(B,R,H) if inverse is not None else values
                return (restored*weights.unsqueeze(-1)).sum(1)
            def native():
                return reduce(values if kind=="mix" else source,weights,
                              inverse=inverse,mean=kind=="mix",order=orders[kind])
            actual=native();assert same(actual,reference()),(kind,B)
            graph=torch.npu.NPUGraph()
            with torch.npu.graph(graph):out=native()
            for _ in range(4):
                values.copy_(torch.randn_like(values)*.1);weights.copy_(torch.rand_like(weights))
                graph.replay()
                assert same(out,reference()),("dynamic",kind,B)
            del graph,out
            repetitions=32 if B==1 else 1
            row=dict(kind=kind,batch=B,order=orders[kind],exact=True,
                     reference_ms=benchmark(reference,repetitions),
                     native_ms=benchmark(native,repetitions))
            print("WEIGHTED_REDUCE",json.dumps(row),flush=True);records.append(row)
            del values,weights,inverse,source,actual
            gc.collect();torch.npu.empty_cache()
    Path(os.environ.get("FNX_REDUCE_TEST_OUTPUT","/tmp/fnx-reduce-test.json")).write_text(json.dumps(dict(orders=orders,records=records),indent=2))
    print("R118_REDUCE_ACCEPT",flush=True)


if __name__=="__main__":main()
