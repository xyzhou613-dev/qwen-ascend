"""Incremental pooled keys must preserve index selection exactly."""
import os,gc,time
import torch
from fnx import engine as E
from fnx.load import load_layer,QDIR

@torch.no_grad()
def main():
    dev='npu:0';torch.npu.set_device(dev)
    W=load_layer(os.path.join(QDIR,'L03.safetensors'),3,dev)
    L=E.FnLayer(3,'full',W,dev);del W
    L.kv=E.KVStore(262144,dev,1)
    L.index_k=torch.randn(262144,128,device=dev,dtype=torch.bfloat16)*.03
    for T,start in ((8,2040),(8,262136),(1,210001),(32,210003),(7,2045)):
        h=torch.randn(T,1,E.H,device=dev,dtype=torch.bfloat16)*.03
        pos=torch.arange(start,start+T,device=dev)
        os.environ['FNX_INDEX_CACHE']='0';a,av=L._indexer_indices(h,pos)
        # Simulate a loaded prefix as well as subsequent incremental updates.
        L.index_keys=None
        os.environ['FNX_INDEX_CACHE']='1';b,bv=L._indexer_indices(h,pos)
        assert torch.equal(a,b) and torch.equal(av,bv)
        pos=pos-1
        os.environ['FNX_INDEX_CACHE']='0';a,av=L._indexer_indices(h,pos)
        os.environ['FNX_INDEX_CACHE']='1';b,bv=L._indexer_indices(h,pos)
        assert torch.equal(a,b) and torch.equal(av,bv)
        print('INDEX_CACHE_EXACT',T,start,flush=True)
    print('INDEX_CACHE_ACCEPT',flush=True)

if __name__=='__main__':main()
