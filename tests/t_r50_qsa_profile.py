"""Locate the long-prefill attention hotspot and check grouped-head BMM."""
import os,time,gc
import torch
from fnx import engine as E
from fnx.load import load_layer,QDIR
from t_r50_layer_profile import measure


def main():
    torch.npu.set_device(0)
    W=load_layer(os.path.join(QDIR,'L03.safetensors'),3,'npu:0')
    L=E.FnLayer(3,'full',W,'npu:0'); del W
    L.kv=E.KVStore(8192,'npu:0',1)
    L.index_k=torch.randn(8192,128,dtype=torch.bfloat16,device='npu:0')*.02
    for T in (8,32,64):
        h=torch.randn(T,1,E.H,dtype=torch.bfloat16,device='npu:0')*.02
        p=torch.tensor([4096],device='npu:0',dtype=torch.int64)
        pos=p+torch.arange(T,device='npu:0')
        a=measure(lambda:L._indexer_mask(h,pos))
        L.index_enabled=False
        b=measure(lambda:L._qsa_chunk(h,p,T))
        L.index_enabled=True
        q=torch.randn(T,4,128,dtype=torch.float32,device='npu:0')
        k=torch.randn(128,2048,dtype=torch.float32,device='npu:0')
        old=measure(lambda:torch.matmul(q,k))
        new=measure(lambda:torch.matmul(q.reshape(T*4,128),k).view(T,4,2048))
        d=(torch.matmul(q,k)-torch.matmul(q.reshape(T*4,128),k).view(T,4,2048)).abs()
        print('QSA_PART',T,{'indexer_ms':a,'attention_no_index_ms':b,'score_old':old,'score_flat':new,'score_diff':float(d.max())},flush=True)


if __name__=='__main__':
    with torch.no_grad(): main()
