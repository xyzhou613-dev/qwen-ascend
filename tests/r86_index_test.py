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
    for cap in (16384,32768,65536,98304,131072,196608):
        for T,end in ((1,cap+1),(1,cap+2),(1,cap+3),(128,cap+3)):
            h=torch.randn(T,1,E.H,dtype=torch.bfloat16,device='npu:0')*.03
            pos=torch.arange(end-T,end,device=h.device)
            L.index_scan_cap=262144;a,av=L._indexer_indices(h,pos)
            L.index_scan_cap=cap;b,bv=L._indexer_indices(h,pos)
            assert torch.equal(a,b) and torch.equal(av,bv),(cap,T,end)
        print('COMPLETE_GROUP_EXACT',cap,flush=True)
    print('COMPLETE_GROUP_ACCEPT',flush=True)

if __name__=='__main__':main()
