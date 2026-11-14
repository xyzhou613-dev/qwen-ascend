"""Experimental independent-row decode. Never imported by production r152.

One dense projection for B rows and one GMM with B*TOPK selected groups;
no loop over request forwards. Chunk prefill retains the original path.
"""
import torch
import torch_npu
import write_nd as ND
from fnx import engine as E
F=E.F
ORIGINAL_MOE=E.FnLayer._moe
ORIGINAL_QSA=E.FnLayer._qsa
AUDIT={}

def selected(exp,x,ids):
    assert exp.expert_major and exp._scales is not None
    ends=torch.arange(1,ids.numel()+1,device=x.device,dtype=torch.int64)
    return torch_npu.npu_grouped_matmul(
        [x.contiguous()],[exp.w4.index_select(0,ids)],
        antiquant_scale=[exp._scales.index_select(0,ids)],
        group_list=ends,split_item=3,group_type=0,
        output_dtype=torch.bfloat16)[0]

def moe(self,x):
    b=x.shape[0]
    if b>4:return ORIGINAL_MOE(self,x)
    AUDIT['moe_rows']=b;AUDIT['gmm_groups']=b*E.TOPK
    x=x.reshape(b,E.H)
    if self.sh_gu is not None:
        gu=E.lin(x,self.sh_gu)
        sh=E.lin(F.silu(gu[:,:E.MI])*gu[:,E.MI:2*E.MI],self.sh_down)
        if self.sh_gw:
            sh=torch.sigmoid(gu[:,2*E.MI:2*E.MI+1])*sh
            logits=gu[:,2*E.MI+1:]
        else:
            sh=torch.sigmoid(E.lin(x,self.sh_gate2))*sh
            logits=E.lin(x,self.gate_w)
    else:
        sh=E.lin(F.silu(E.lin(x,self.sh_gate))*E.lin(x,self.sh_up),self.sh_down)
        sh=torch.sigmoid(E.lin(x,self.sh_gate2))*sh
        logits=E.lin(x,self.gate_w)
    probs=torch.softmax(logits.float(),-1)
    w,idx=torch.topk(probs,E.TOPK,dim=-1)
    w=(w/w.sum(-1,keepdim=True)).to(torch.bfloat16)
    ids=idx.reshape(-1)
    gu=selected(self.exp_gu,x.repeat_interleave(E.TOPK,0),ids)
    act=(F.silu(gu[:,:E.MI].float())*gu[:,E.MI:].float()).to(torch.bfloat16)
    yd=selected(self.exp_dn,act,ids).reshape(b,E.TOPK,E.H)
    return ((yd*w.unsqueeze(-1)).sum(1)+sh).reshape(b,1,E.H)

def prepare(layer):
    if layer.ltype!='full':return
    b=layer.conv.shape[0];p=layer.kv.P
    layer.index_k=torch.zeros(b,p,128,device=layer.dev,dtype=torch.bfloat16)
    layer.index_keys=torch.zeros(b,p//4,128,device=layer.dev,dtype=torch.float32)

def indices(self,h,pos):
    b=h.shape[0];p=self.kv.P;groups=p//4
    qk=E.lin(h,self.index_w).reshape(b,640)
    query=E.rms_norm(qk[:,:512].reshape(b,4,128),self.index_qn)
    cs,sn=E.rope_cs(pos,self.inv)
    query=E.apply_rope(query,cs[:,None,:],sn[:,None,:])
    raw=qk[:,512:]
    ND.index(self.index_k,raw,pos)
    gid=pos//4
    offsets=gid[:,None]*4+torch.arange(4,device=h.device)[None,:]
    pooled=self.index_k.gather(1,offsets[:,:,None].expand(b,4,128)).float().mean(1).to(torch.bfloat16)
    pooled=E.rms_norm(pooled,self.index_kn)
    cs,sn=E.rope_cs(gid*4,self.inv)
    pooled=E.apply_rope(pooled,cs,sn)
    ND.index(self.index_keys,pooled.float(),gid)
    scores=torch.bmm(query.float(),self.index_keys.transpose(1,2))
    scores=F.relu(scores).sum(1)*(128**-.5)
    starts=torch.arange(groups,device=h.device)[None,:]*4
    scores=scores.masked_fill(starts+4>pos[:,None]+1,E.NEG)
    chosen=torch.topk(scores,min(512,groups),dim=-1).indices
    rows=(chosen[:,:,None]*4+torch.arange(4,device=h.device)).reshape(b,-1)
    full=((pos+1)//4)*4
    valid=rows<full[:,None]
    tail=full[:,None]+torch.arange(3,device=h.device)[None,:]
    rows=torch.cat((rows,tail),1)
    valid=torch.cat((valid,tail<=pos[:,None]),1)
    # Preserve physical token order for attention reduction.
    rows=torch.where(valid,rows,p).float().sort(-1).values.to(torch.int64)
    valid=rows<p
    return rows.clamp_max(p-1),valid

def qsa(self,h,pos):
    b=h.shape[0];AUDIT['qsa_rows']=b
    cs,sn=E.rope_cs(pos,self.inv)
    if self.w_qkvf is not None:
        qkv=E.lin(h,self.w_qkvf)[:,0];nq=E.NAH*2*E.HD
        qg=qkv[:,:nq].reshape(b,E.NAH,2*E.HD)
        k=E.rms_norm(qkv[:,nq:nq+E.KVH*E.HD].reshape(b,E.KVH,E.HD),self.kn_w)
        v=qkv[:,nq+E.KVH*E.HD:].reshape(b,E.KVH,E.HD)
    else:
        qg=E.lin(h,self.w_q).reshape(b,E.NAH,2*E.HD)
        k=E.rms_norm(E.lin(h,self.w_k).reshape(b,E.KVH,E.HD),self.kn_w)
        v=E.lin(h,self.w_v).reshape(b,E.KVH,E.HD)
    q=E.apply_rope(E.rms_norm(qg[...,:E.HD],self.qn_w),cs[:,None,:],sn[:,None,:])
    k=E.apply_rope(k,cs[:,None,:],sn[:,None,:])
    kc,vc=self.kv.k[self.ki],self.kv.v[self.ki]
    where=pos[:,None,None,None].expand(b,E.KVH,1,E.HD)
    ND.kv(kc,k,pos);ND.kv(vc,v,pos)
    rows,valid=indices(self,h,pos);width=rows.shape[1]
    ix=rows[:,None,:,None].expand(b,E.KVH,width,E.HD)
    kk=kc.gather(2,ix).reshape(b*E.KVH,width,E.HD)
    vv=vc.gather(2,ix).reshape(b*E.KVH,width,E.HD)
    qq=q.reshape(b*E.KVH,E.NAH//E.KVH,E.HD)
    score=torch.bmm(qq,kk.transpose(1,2))*(E.HD**-.5)
    mask=valid[:,None,None,:].expand(b,E.KVH,1,width).reshape(b*E.KVH,1,width)
    att=torch.softmax(score.masked_fill(~mask,E.NEG).float(),-1).to(torch.bfloat16)
    out=torch.bmm(att,vv).reshape(b,E.NAH*E.HD)
    return E.lin(out*torch.sigmoid(qg[...,E.HD:].reshape(b,-1)),self.w_o)

def install():
    E.FnLayer._moe=moe
    E.FnLayer._qsa=qsa
