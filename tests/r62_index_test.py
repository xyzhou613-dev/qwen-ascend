"""Dropping future masked index keys must preserve selected ids and masks."""
import os
import torch
from fnx import engine as E
from fnx.load import load_layer,QDIR

@torch.no_grad()
def main():
    torch.npu.set_device(0)
    W=load_layer(os.path.join(QDIR,'L03.safetensors'),3,'npu:0')
    L=E.FnLayer(3,'full',W,'npu:0');del W
    L.kv=E.KVStore(262144,'npu:0',1)
    L.index_k=torch.randn(262144,128,dtype=torch.bfloat16,device='npu:0')*.03
    for T,start in ((1,2048),(128,4096),(128,16256),(1,16383),(128,16384),
                    (128,65408),(1,65535),(128,65536),(128,130944),
                    (1,131071),(128,131072),(128,262016)):
        h=torch.randn(T,1,E.H,dtype=torch.bfloat16,device='npu:0')*.03
        pos=torch.arange(start,start+T,device='npu:0')
        L.index_scan_cap=262144;a,av=L._indexer_indices(h,pos)
        cap=min(c for c in (16384,65536,131072,262144) if c>=start+T)
        L.index_scan_cap=cap;b,bv=L._indexer_indices(h,pos)
        assert torch.equal(a,b) and torch.equal(av,bv),(T,start,cap)
        print('INDEX_BUCKET_EXACT',T,start,cap,flush=True)
    print('INDEX_BUCKET_ACCEPT',flush=True)

if __name__=='__main__':main()
