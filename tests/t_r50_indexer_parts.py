"""Captured micro-costs for every stage of QSA block selection."""
import os,math
import torch
from fnx import engine as E
from fnx.load import load_layer,QDIR
from t_r50_layer_profile import measure


def main():
    torch.npu.set_device(0)
    W=load_layer(os.path.join(QDIR,'L03.safetensors'),3,'npu:0')
    L=E.FnLayer(3,'full',W,'npu:0'); del W
    P=8192; G=P//4
    L.kv=E.KVStore(P,'npu:0',1)
    L.index_k=torch.randn(P,128,dtype=torch.bfloat16,device='npu:0')*.02
    for T in (8,64):
        h=torch.randn(T,1,E.H,dtype=torch.bfloat16,device='npu:0')*.02
        pos=torch.arange(4096,4096+T,device='npu:0')
        qk=E.lin(h,L.index_w).reshape(T,640)
        q=E.rms_norm(qk[:,:512].reshape(T,4,128),L.index_qn)
        starts=torch.arange(G,device='npu:0')*4
        pooled=L.index_k.view(G,4,128).float().mean(1).to(h.dtype)
        keys=E.rms_norm(pooled,L.index_kn)
        scores=(q.float()@keys.float().T).relu().sum(1)/math.sqrt(128)
        complete=starts[None,:]+4<=pos[:,None]+1
        scores=scores.masked_fill(~complete,E.NEG)
        chosen=scores.topk(512,dim=-1).indices
        blocks=torch.zeros_like(scores,dtype=torch.bool).scatter_(1,chosen,True)&complete
        raw=qk[:,512:].contiguous()
        operations={
            'proj':lambda:E.lin(h,L.index_w),
            'index_copy':lambda:L.index_k.index_copy_(0,pos,raw),
            'pool':lambda:L.index_k.view(G,4,128).float().mean(1).to(h.dtype),
            'norm':lambda:E.rms_norm(pooled,L.index_kn),
            'score':lambda:(q.float()@keys.float().T).relu().sum(1),
            'topk':lambda:scores.topk(512,dim=-1),
            'topk_unsorted':lambda:scores.topk(512,dim=-1,sorted=False),
            'sort':lambda:scores.sort(dim=-1,descending=True),
            'scatter_bool':lambda:torch.zeros_like(scores,dtype=torch.bool).scatter_(1,chosen,True),
            'repeat':lambda:blocks.repeat_interleave(4,dim=-1),
            'expand_repeat':lambda:blocks.unsqueeze(-1).expand(T,G,4).reshape(T,G*4),
            'select_repeat':lambda:blocks.index_select(1,torch.arange(G*4,device='npu:0')//4),
            'scatter_float':lambda:torch.zeros_like(scores).scatter_(1,chosen,1.),
        }
        for name,fn in operations.items():
            print('INDEX_PART',T,name,measure(fn),flush=True)
        alt=scores.topk(512,dim=-1,sorted=False).indices
        alt2=scores.sort(dim=-1,descending=True).indices[:,:512]
        to_mask=lambda ids:torch.zeros_like(scores,dtype=torch.bool).scatter_(1,ids,True)
        print('MASK_DIFF',T,int((to_mask(alt)!=to_mask(chosen)).sum()),int((to_mask(alt2)!=to_mask(chosen)).sum()),flush=True)


if __name__=='__main__':
    with torch.no_grad():main()
