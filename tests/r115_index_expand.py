import os
import json
import statistics
import time
from pathlib import Path
import torch
import torch_npu
from fnx.index_expand import expand

P=262144
device="npu:0"


def reference(chosen,pos):
    T=chosen.shape[0]
    tokens=(chosen[:,:,None]*4+torch.arange(4,device=device)).reshape(T,-1)
    valid=tokens<((pos[:,None]+1)//4)*4
    tail=((pos[:,None]+1)//4)*4+torch.arange(3,device=device)
    tokens=torch.cat((tokens,tail),-1)
    valid=torch.cat((valid,tail<=pos[:,None]),-1)
    ordered=torch.where(valid,tokens,P).float().sort(-1).values.to(torch.int64)
    return ordered.clamp_max(P-1),ordered<P


def inputs(T,start):
    pos=(torch.arange(T,device=device)+start).clamp_max(P-1)
    scores=torch.rand(T,65536,device=device)
    groups=torch.arange(65536,device=device)[None,:]
    scores.masked_fill_(groups>=(pos[:,None]+1)//4,-1.0e30)
    chosen=scores.topk(512,-1).indices
    return chosen,pos


def equal(a,b):
    return all(torch.equal(x,y) for x,y in zip(a,b))


def benchmark(fn):
    graph=torch.npu.NPUGraph()
    torch.npu.synchronize()
    with torch.npu.graph(graph):
        for _ in range(16):out=fn()
    graph.replay();torch.npu.synchronize()
    times=[]
    for _ in range(10):
        start=time.perf_counter();graph.replay();torch.npu.synchronize()
        times.append((time.perf_counter()-start)*1000/16)
    return statistics.median(times[2:])


@torch.no_grad()
def main():
    torch.manual_seed(115)
    rows=[]
    for T in (1,17,128,512):
        chosen,pos=inputs(T,196608)
        expand(chosen,pos,P)
        graph=torch.npu.NPUGraph()
        with torch.npu.graph(graph):actual=expand(chosen,pos,P)
        for start in (0,3,4,1539,2047,2048,65535,196608,P-T):
            new,p=inputs(T,start)
            chosen.copy_(new);pos.copy_(p)
            graph.replay()
            assert equal(actual,reference(chosen,pos)),(T,start)
        row=dict(tokens=T,exact=True,reference_ms=benchmark(lambda:reference(chosen,pos)),
                 native_ms=benchmark(lambda:expand(chosen,pos,P)))
        rows.append(row);print("INDEX_EXPAND",json.dumps(row),flush=True)
    Path(os.environ.get("FNX_INDEX_EXPAND_OUTPUT", "/tmp/fnx-index-expand.json")).write_text(json.dumps(rows,indent=2))
    print('R115_EXPAND_ACCEPT',flush=True)


if __name__=='__main__':main()
