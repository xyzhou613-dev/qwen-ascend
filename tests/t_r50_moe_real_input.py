"""MoE on the first layer's actual activations; test empty-group handling."""
import os
import torch
import torch_npu
from transformers import AutoTokenizer
from fnx import engine as E
from fnx.model import FnModel


def main():
    os.environ['FNX_MOE_GROUP']='1'
    m=FnModel(P=8192,lpc=1,dev0='npu:0',dev1='npu:1',nlayers=1,with_head=False,with_ngram=False,verbose=False)
    tk=AutoTokenizer.from_pretrained('/models/Qwen3.8-Flash-Next')
    ids=tk('昇腾处理器面向大规模深度学习训练与推理场景。').input_ids[:8]
    L=m.layers0[0];T=len(ids)
    h=m.embed_rows(ids).to(m.dev0).view(T,1,E.H).repeat(1,1,E.HC)
    saved=L._moe; captured=[]
    def hook(x):captured.append(x.clone());return saved(x)
    L._moe=hook;os.environ['FNX_MOE_GROUP']='0';os.environ['FNX_PREFILL_DN']='torch'
    L.forward_chunk(h,0,T);L._moe=saved
    x=torch.cat(captured,0)
    ref=torch.cat([L._moe(x[t:t+1]) for t in range(T)],0)
    print('INPUT',x.abs().mean().item(),x.abs().max().item(),flush=True)
    os.environ['FNX_MOE_GROUP']='1'
    for split in ('0','32','64'):
        os.environ['FNX_MOE_GROUP_SPLIT']=split
        got=L._moe(x)
        d=(got.float()-ref.float()).abs()
        print('REAL_MOE',split,'rel',float(d.mean()/ref.float().abs().mean()),'max',float(d.max()),flush=True)
    x2=x.view(T,E.H)
    logits=E.lin(x2,L.sh_gu)[:,2*E.MI+1:]
    probs=torch.softmax(logits.float(),-1)
    w,idx=probs.topk(E.TOPK,-1);w=(w/w.sum(-1,keepdim=True)).to(torch.bfloat16)
    ids=idx.flatten();order=ids.float().argsort();counts=torch.bincount(ids,minlength=512)
    nz=torch.nonzero(counts).flatten();ends=counts[nz].cumsum(0)
    rows=x2.index_select(0,order//E.TOPK)
    def compact(exp,inp):
        scales=exp.s8t.index_select(1,nz).permute(1,0,2).to(torch.bfloat16)*exp.base[nz,None,:]
        return torch_npu.npu_grouped_matmul([inp],[exp.w4.index_select(0,nz)],
               antiquant_scale=[scales],group_list=ends,group_type=0,split_item=3,output_dtype=torch.bfloat16)[0]
    gu=compact(L.exp_gu,rows)
    act=(torch.nn.functional.silu(gu[:,:E.MI].float())*gu[:,E.MI:].float()).to(torch.bfloat16)
    yd=compact(L.exp_dn,act)
    restored=torch.empty_like(yd).index_copy_(0,order,yd)
    expert=(restored.view(T,E.TOPK,E.H)*w[:,:,None]).sum(1)
    sh=E.lin(x2,L.sh_gu)
    shared=E.lin(torch.nn.functional.silu(sh[:,:E.MI])*sh[:,E.MI:2*E.MI],L.sh_down)*torch.sigmoid(sh[:,2*E.MI:2*E.MI+1])
    got=(expert+shared).view(T,1,E.H)
    d=(got.float()-ref.float()).abs()
    print('COMPACT',len(nz),float(d.mean()/ref.float().abs().mean()),float(d.max()),'counts',counts[nz].tolist(),flush=True)


if __name__=='__main__':
    with torch.no_grad():main()
