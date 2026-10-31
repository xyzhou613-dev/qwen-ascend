"""Embed GQA K/V in MLA's latent cache without changing attention scores.

The query is [q_nonrope(192), zeros(320)], and the latent cache is
[k_nonrope(192), v(256), zeros(64)]. RoPE's 64 dimensions remain separate.
The native MLA output's [192:448] columns are the original value result.
"""
import torch
import torch.nn.functional as F
import torch_npu

def write_and_view(store,ki,k,v,pos,P,short):
    kc,rc=store.k[ki][:,:,:P],store.v[ki][:,:,:P]
    latent=F.pad(torch.cat((k[...,64:],v),-1),(0,64))
    for head in range(2):
        kc[0,head].index_copy_(0,pos,latent[:,head].contiguous())
        rc[0,head].index_copy_(0,pos,k[:,head,:64].contiguous())
    if short:
        return torch.cat((rc,kc[...,:192]),-1),kc[...,192:448]
    return kc,rc

def attention(q,ids,valid,kc,rc):
    T=q.shape[0];P=kc.shape[2]
    indices=torch.where(valid,ids,-1).to(torch.int32)
    width=(indices.shape[-1]+63)//64*64
    indices=F.pad(indices,(0,width-indices.shape[-1]),value=-1).view(1,T,1,width)
    outputs=[]
    for head in range(2):
        group=q[:,head*12:(head+1)*12]
        query=F.pad(group[...,64:],(0,320,0,4)).view(1,T,16,512)
        rope=F.pad(group[...,:64],(0,0,0,4)).view(1,T,16,64)
        latent=kc[0,head].view(1,P,1,512)
        keyrope=rc[0,head].view(1,P,1,64)
        out=torch_npu.npu_sparse_flash_attention(query,latent,latent,indices,256**-.5,
            sparse_block_size=1,query_rope=rope,key_rope=keyrope,
            layout_query='BSND',layout_kv='BSND',sparse_mode=0,attention_mode=2)[0]
        outputs.append(out[0,:,:12,192:448])
    return torch.cat(outputs,1).reshape(T,24*256)
