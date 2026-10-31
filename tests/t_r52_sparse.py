"""Sparse selected-KV attention against the dense masked reference."""
import os,gc
import torch
from fnx import engine as E
from fnx.load import load_layer,QDIR

def main():
    torch.npu.set_device(0)
    W=load_layer(os.path.join(QDIR,'L03.safetensors'),3,'npu:0')
    L=E.FnLayer(3,'full',W,'npu:0');del W
    for P in (8192,32768,65536,131072):
        L.kv=E.KVStore(P,'npu:0',1)
        L.index_k=torch.randn(P,128,dtype=torch.bfloat16,device='npu:0')*.03
        L.kv.k[0].normal_(0,.03);L.kv.v[0].normal_(0,.03)
        T=8;h=torch.randn(T,1,E.H,dtype=torch.bfloat16,device='npu:0')*.3
        for start in (2040,P-T):
            pos=torch.arange(start,start+T,device='npu:0')
            mask=L._indexer_mask(h,pos)
            ids,valid=L._indexer_indices(h,pos)
            rebuilt=torch.zeros(T,P,dtype=torch.int32,device='npu:0').scatter_add_(1,ids,valid.to(torch.int32))>0
            assert torch.equal(mask,rebuilt),(P,start,int((mask!=rebuilt).sum()))
            os.environ['FNX_QSA_SPARSE']='0'
            ref=L._qsa_chunk(h,start,T)
            os.environ['FNX_QSA_SPARSE']='1'
            got=L._qsa_chunk(h,start,T)
            diff=(ref.float()-got.float()).abs()
            rel=float(diff.mean()/(ref.float().abs().mean()+1e-9))
            print('SPARSE',P,start,'mask_exact',True,'rel',rel,'max',float(diff.max()),flush=True)
            assert torch.isfinite(got).all() and rel<.01
        del mask,rebuilt,ref,got,ids,valid
        L.kv=None;L.index_k=None;gc.collect();torch.npu.empty_cache()
    print('SPARSE_ACCEPT',flush=True)

if __name__=='__main__':
    with torch.no_grad():main()
