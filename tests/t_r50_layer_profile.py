"""Captured real-weight prefill costs, one layer on an exclusive device."""
import os,time,gc
import torch
from fnx import engine as E
from fnx.load import load_layer,QDIR


def measure(fn):
    stream=torch.npu.Stream()
    with torch.npu.stream(stream):
        for _ in range(2): fn()
        torch.npu.synchronize()
        graph=torch.npu.NPUGraph()
        with torch.npu.graph(graph,capture_error_mode='relaxed'): fn()
    torch.npu.synchronize()
    start=time.perf_counter()
    for _ in range(10): graph.replay()
    torch.npu.synchronize()
    result=(time.perf_counter()-start)*100
    del graph
    gc.collect(); torch.npu.empty_cache()
    return result


def main():
    torch.npu.set_device(0)
    for i,typ in ((0,'linear'),(3,'full')):
        W=load_layer(os.path.join(QDIR,f'L{i:02d}.safetensors'),i,'npu:0')
        L=E.FnLayer(i,typ,W,'npu:0'); del W
        if typ=='full':
            L.kv=E.KVStore(8192,'npu:0',1)
            L.index_k=torch.zeros(8192,128,dtype=torch.bfloat16,device='npu:0')
        for T in (8,32,64):
            h=torch.randn(T,1,E.H,dtype=torch.bfloat16,device='npu:0')*.02
            hyper=h.repeat(1,1,E.HC)
            pos=torch.tensor([4096],dtype=torch.int64,device='npu:0')
            attention=(lambda:L._deltanet_chunk(h,T)) if typ=='linear' else (lambda:L._qsa_chunk(h,pos,T))
            att=measure(attention)
            moe=measure(lambda:torch.cat([L._moe(h[t:t+1]) for t in range(T)],0))
            full=measure(lambda:L.forward_chunk(hyper,pos,T))
            print('PROFILE',typ,T,{'attention_ms':att,'moe_ms':moe,'full_ms':full,
                                  'full_ms_per_token':full/T},flush=True)
        del L; gc.collect(); torch.npu.empty_cache()


if __name__=='__main__':
    with torch.no_grad(): main()
