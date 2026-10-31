"""History/query mutation must change captured indexer selection correctly."""
import os
import torch
from fnx import engine as E
from fnx.load import load_layer,QDIR

@torch.no_grad()
def main():
    torch.npu.set_device(0)
    x=torch.ones(1,device='npu:0');g0=torch.npu.NPUGraph()
    with torch.npu.graph(g0,capture_error_mode='relaxed'):x.add_(1)
    dev='npu:1';torch.npu.set_device(dev)
    W=load_layer(os.path.join(QDIR,'L27.safetensors'),27,dev)
    L=E.FnLayer(27,'full',W,dev);del W
    L.kv=E.KVStore(262144,dev,1)
    L.index_k=torch.randn(262144,128,device=dev,dtype=torch.bfloat16)*.03
    for T in (1,128):
        h=torch.randn(T,1,E.H,device=dev,dtype=torch.bfloat16)*.1
        pos=200000+torch.arange(T,device=dev)
        a,v=L._indexer_indices(h,pos);ob=torch.zeros_like(a);vb=torch.zeros_like(v)
        def fn():
            a,v=L._indexer_indices(h,pos);ob.copy_(a);vb.copy_(v)
        s=torch.npu.Stream()
        with torch.npu.stream(s):
            fn();torch.npu.synchronize(dev);g=torch.npu.NPUGraph()
            with torch.npu.graph(g,stream=s,capture_error_mode='relaxed'):fn()
        for trial in range(3):
            L.index_k.normal_(0,.1);h.normal_(0,.2);pos.add_(trial+1);torch.npu.synchronize(dev)
            ref,valid=L._indexer_indices(h,pos);torch.npu.synchronize(dev)
            with torch.npu.stream(s):g.replay()
            torch.npu.synchronize(dev)
            exact=torch.equal(ob,ref) and torch.equal(vb,valid)
            print('INDEX_GRAPH',T,trial,exact,int((ob!=ref).sum()),flush=True)
            assert exact
    print('INDEX_GRAPH_ACCEPT',flush=True)

if __name__=='__main__':main()
