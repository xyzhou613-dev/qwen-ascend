"""Real L27 QSA replay versus eager after changing history and queries."""
import torch
from fnx import engine as E
from fnx.load import load_layer,QDIR
import os

@torch.no_grad()
def main():
    torch.npu.set_device(0);x=torch.ones(1,device='npu:0');g0=torch.npu.NPUGraph()
    with torch.npu.graph(g0,capture_error_mode='relaxed'):x.add_(1)
    dev='npu:1';torch.npu.set_device(dev)
    W=load_layer(os.path.join(QDIR,'L27.safetensors'),27,dev)
    L=E.FnLayer(27,'full',W,dev);del W
    L.kv=E.KVStore(262144,dev,1)
    L.index_k=torch.randn(262144,128,device=dev,dtype=torch.bfloat16)*.03
    for T in (1,128):
        h=torch.randn(T,1,E.H,device=dev,dtype=torch.bfloat16)*.1
        pos=torch.tensor([200000],device=dev)
        fn=(lambda:L._qsa(h,pos)) if T==1 else (lambda:L._qsa_chunk(h,pos,T))
        out=fn();buf=torch.zeros_like(out)
        s=torch.npu.Stream()
        with torch.npu.stream(s):
            fn();torch.npu.synchronize(dev);g=torch.npu.NPUGraph()
            with torch.npu.graph(g,stream=s,capture_error_mode='relaxed'):buf.copy_(fn())
        for trial in range(3):
            L.index_k.normal_(0,.1);L.kv.k[0].normal_(0,.1);L.kv.v[0].normal_(0,.1)
            h.normal_(0,.2);pos.add_(trial+1);torch.npu.synchronize(dev)
            ref=fn();torch.npu.synchronize(dev)
            with torch.npu.stream(s):g.replay()
            torch.npu.synchronize(dev)
            rel=float((ref.float()-buf.float()).abs().mean()/(ref.float().abs().mean()+1e-9))
            print('QSA_GRAPH',T,trial,rel,float((ref.float()-buf.float()).abs().max()),flush=True)
            assert rel<1e-5
    print('QSA_GRAPH_ACCEPT',flush=True)

if __name__=='__main__':main()
