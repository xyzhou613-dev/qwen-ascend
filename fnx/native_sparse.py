"""Embed GQA K/V in MLA's latent cache without changing attention scores.

The query is [q_nonrope(192), zeros(320)], and the latent cache is
[k_nonrope(192), v(256), zeros(64)]. RoPE's 64 dimensions remain separate.
The native MLA output's [192:448] columns are the original value result.
"""
import torch
import torch.nn.functional as F
import torch_npu

def write_and_view(store,ki,k,v,pos,P,short):
    if getattr(store,"native_dual",False):
        original_k,original_v=store.k[ki][:,:,:P],store.v[ki][:,:,:P]
        original_k[0].transpose(0,1).index_copy_(0,pos,k.contiguous())
        original_v[0].transpose(0,1).index_copy_(0,pos,v.contiguous())
        rows=(pos[None,:]+store.head_offsets).reshape(-1)
        latent=F.pad(torch.cat((k[...,64:],v),-1),(0,64))
        store.k[ki+store.n].view(2*store.P,512).index_copy_(0,rows,latent.transpose(0,1).contiguous().view(-1,512))
        store.v[ki+store.n].view(2*store.P,64).index_copy_(0,rows,k[...,:64].transpose(0,1).contiguous().view(-1,64))
        if short or k.shape[0]==1:return original_k,original_v
        return store.k[ki+store.n][:,:,:P],store.v[ki+store.n][:,:,:P]
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
    if T == 1 and kc.shape[-1]==256:
        width=ids.shape[-1]
        keys=kc[0].transpose(0,1).index_select(0,ids.reshape(-1)).view(1,width,2,256)
        values=rc[0].transpose(0,1).index_select(0,ids.reshape(-1)).view(1,width,2,256)
        out=torch_npu.npu_fused_infer_attention_score(q.view(1,1,24,256),keys,values,
            num_heads=24,num_key_value_heads=2,input_layout='BSND',scale=256**-.5,
            atten_mask=(~valid).view(1,1,1,width))[0]
        return out.view(1,24*256)
    if T == 1:
        # Keep decode on the compact GQA kernel. One gather per packed cache
        # uses flattened (head,position) rows, avoiding padded query heads.
        width=ids.shape[-1]
        rows=(ids.reshape(1,-1)+torch.arange(2,device=ids.device)[:,None]*P).reshape(-1)
        latent=kc.reshape(2*P,512).index_select(0,rows).view(2,width,512).transpose(0,1)
        rope=rc.reshape(2*P,64).index_select(0,rows).view(2,width,64).transpose(0,1)
        keys=torch.cat((rope,latent[...,:192]),-1).view(1,width,2,256)
        values=latent[...,192:448].contiguous().view(1,width,2,256)
        out=torch_npu.npu_fused_infer_attention_score(q.view(1,1,24,256),keys,values,
            num_heads=24,num_key_value_heads=2,input_layout='BSND',scale=256**-.5,
            atten_mask=(~valid).view(1,1,1,width))[0]
        return out.view(1,24*256)
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
