"""Validate packed cache writes and exact reconstruction for short attention."""
import os
import torch
from fnx import engine as E
from fnx.native_sparse import write_and_view,attention

@torch.no_grad()
def main():
    torch.npu.set_device(0)
    store=E.KVStore(262144,'npu:0',1)
    assert store.native_sparse
    for T,start in ((8,2040),(1,210001),(128,260000)):
        k=torch.randn(T,2,256,dtype=torch.bfloat16,device='npu:0')
        v=torch.randn_like(k);pos=torch.arange(start,start+T,device=k.device)
        kc,rc=write_and_view(store,0,k,v,pos,store.P,False)
        for head in range(2):
            rows=kc[0,head].index_select(0,pos)
            assert torch.equal(rows[:,:192],k[:,head,64:])
            assert torch.equal(rows[:,192:448],v[:,head])
            assert torch.equal(rc[0,head].index_select(0,pos),k[:,head,:64])
        print('NATIVE_CACHE_EXACT',T,start,flush=True)
    k=torch.randn(8,2,256,dtype=torch.bfloat16,device='npu:0');v=torch.randn_like(k)
    pos=torch.arange(16,24,device=k.device)
    kc,vc=write_and_view(store,0,k,v,pos,2048,True)
    assert torch.equal(kc[0].index_select(1,pos),k.transpose(0,1))
    assert torch.equal(vc[0].index_select(1,pos),v.transpose(0,1))
    print('NATIVE_CACHE_ACCEPT',flush=True)

if __name__=='__main__':main()
