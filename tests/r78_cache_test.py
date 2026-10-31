import torch,torch_npu
from fnx import engine as E
from fnx.native_sparse import write_and_view,attention

@torch.no_grad()
def main():
    torch.npu.set_device(0);m=E.KVStore(262144,'npu:0',1)
    assert m.native_dual and len(m.k)==2 and len(m.v)==2
    for T,start in ((8,2040),(1,210001),(128,260000)):
        k=torch.randn(T,2,256,dtype=torch.bfloat16,device='npu:0');v=torch.randn_like(k)
        pos=torch.arange(start,start+T,device=k.device)
        write_and_view(m,0,k,v,pos,m.P,False)
        assert torch.equal(m.k[0][0].transpose(0,1).index_select(0,pos),k)
        assert torch.equal(m.v[0][0].transpose(0,1).index_select(0,pos),v)
        for head in range(2):
            packed=m.k[1][0,head].index_select(0,pos)
            assert torch.equal(packed[:,:192],k[:,head,64:])
            assert torch.equal(packed[:,192:448],v[:,head])
            assert torch.equal(m.v[1][0,head].index_select(0,pos),k[:,head,:64])
    q=torch.randn(1,24,256,dtype=k.dtype,device=k.device)
    ids=torch.arange(2048,4099,device=k.device).view(1,-1);valid=torch.ones_like(ids,dtype=torch.bool)
    out=attention(q,ids,valid,m.k[0],m.v[0])
    keys=m.k[0][0].transpose(0,1).index_select(0,ids[0]).view(1,-1,2,256)
    values=m.v[0][0].transpose(0,1).index_select(0,ids[0]).view(1,-1,2,256)
    ref=torch_npu.npu_fused_infer_attention_score(q.view(1,1,24,256),keys,values,
        num_heads=24,num_key_value_heads=2,input_layout='BSND',scale=256**-.5,
        atten_mask=(~valid).view(1,1,1,-1))[0].view(1,-1)
    assert torch.equal(out,ref)
    print('DUAL_CACHE_BITEXACT_ACCEPT',flush=True)

if __name__=='__main__':main()
