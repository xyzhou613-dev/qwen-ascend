"""Position-major BF16 KV and fused selected attention against the old path."""
import os,gc
import torch
from fnx import engine as E
from fnx.load import load_layer,QDIR

@torch.no_grad()
def main():
    dev='npu:0';torch.npu.set_device(dev)
    W=load_layer(os.path.join(QDIR,'L03.safetensors'),3,dev)
    old=E.FnLayer(3,'full',W,dev);new=E.FnLayer(3,'full',W,dev);del W
    os.environ['FNX_KV_POSITION_MAJOR']='0';old.kv=E.KVStore(262144,dev,1)
    os.environ['FNX_KV_POSITION_MAJOR']='1';new.kv=E.KVStore(262144,dev,1)
    old.index_k=torch.randn(262144,128,device=dev,dtype=torch.bfloat16)*.03
    new.index_k=old.index_k.clone()
    for a,b in zip(old.kv.k+old.kv.v,new.kv.k+new.kv.v):
        a.normal_(0,.03);b.copy_(a);assert torch.equal(a,b)
    for T,start in ((8,2040),(1,210001),(128,260000)):
        h=torch.randn(T,1,E.H,device=dev,dtype=torch.bfloat16)*.03
        pos=torch.tensor([start],device=dev)
        os.environ['FNX_SPARSE_FLASH']='0'
        ref=old._qsa(h,pos) if T==1 else old._qsa_chunk(h,start,T)
        os.environ['FNX_SPARSE_FLASH']='1'
        got=new._qsa(h,pos) if T==1 else new._qsa_chunk(h,pos,T)
        rel=float((ref.float()-got.float()).abs().mean()/(ref.float().abs().mean()+1e-9))
        assert torch.equal(old.kv.k[0],new.kv.k[0]) and torch.equal(old.kv.v[0],new.kv.v[0])
        print('LAYOUT_FUSED',T,start,rel,flush=True)
        assert torch.isfinite(got).all() and rel<.01
    old.short_qsa=new.short_qsa=True
    for T,start in ((1,100),(8,1000)):
        h=torch.randn(T,1,E.H,device=dev,dtype=torch.bfloat16)*.03
        pos=torch.tensor([start],device=dev)
        ref=old._qsa(h,pos) if T==1 else old._qsa_chunk(h,pos,T)
        got=new._qsa(h,pos) if T==1 else new._qsa_chunk(h,pos,T)
        print('LAYOUT_SHORT',T,float((ref.float()-got.float()).abs().max()),flush=True)
        assert torch.equal(ref,got)
    print('LAYOUT_FUSED_ACCEPT',flush=True)

if __name__=='__main__':main()
