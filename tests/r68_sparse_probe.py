"""Probe native indexed sparse attention on A2 with production Q/KV shapes."""
import torch,torch_npu,time
import torch.nn.functional as F

@torch.no_grad()
def main():
    torch.npu.set_device(0)
    for T in (1,128):
        P=8192;W=2112
        q=torch.randn(1,T,24,256,dtype=torch.bfloat16,device='npu:0')*.1
        k=torch.randn(1,P,2,256,dtype=q.dtype,device=q.device)*.1;v=torch.randn_like(k)
        ids=torch.arange(W,device=q.device).expand(T,W).contiguous()
        valid=ids<2051
        sparse=torch.where(valid,ids,-1).to(torch.int32)[None,:,None,:].expand(1,T,2,W).contiguous()
        keys=k[0].index_select(0,ids.reshape(-1)).view(T,W,2,256)
        values=v[0].index_select(0,ids.reshape(-1)).view(T,W,2,256)
        expected=torch_npu.npu_fused_infer_attention_score(q.reshape(T,1,24,256),keys,values,
            num_heads=24,num_key_value_heads=2,input_layout='BSND',scale=256**-.5,
            atten_mask=(~valid)[:,None,None,:])[0].view(1,T,24,256)
        def fn():
            pieces=[]
            for head in range(2):
                latent=F.pad(torch.cat((k[:,:,head:head+1,64:],v[:,:,head:head+1]),-1),(0,64)).contiguous()
                a=torch_npu.npu_sparse_flash_attention(F.pad(q[:,:,head*12:(head+1)*12,64:],(0,320,0,4)).contiguous(),
                    latent,latent,
                    sparse[:,:,head:head+1].contiguous(),256**-.5,sparse_block_size=1,
                    query_rope=F.pad(q[:,:,head*12:(head+1)*12,:64],(0,0,0,4)).contiguous(),
                    key_rope=k[:,:,head:head+1,:64].contiguous(),
                    layout_query='BSND',layout_kv='BSND',sparse_mode=0,attention_mode=2)
                pieces.append((a[0] if isinstance(a,(tuple,list)) else a)[:,:,:12,192:448])
            return torch.cat(pieces,2)
        print('SPARSE_START',T,flush=True)
        out=fn();torch.npu.synchronize()
        if isinstance(out,(tuple,list)):out=out[0]
        rel=((out.float()-expected.float()).square().sum()/expected.float().square().sum()).sqrt().item()
        print('SPARSE_ERROR',T,rel,flush=True)
        assert rel<.01
        stream=torch.npu.Stream()
        with torch.npu.stream(stream):
            graph=torch.npu.NPUGraph()
            with torch.npu.graph(graph,stream=stream,capture_error_mode='relaxed'):captured=fn()
        q.normal_(0,.1);k.normal_(0,.1);v.normal_(0,1)
        graph.replay();torch.npu.synchronize()
        eager=fn();torch.npu.synchronize()
        assert torch.equal(captured,eager),'native sparse graph did not track changing inputs'
        del graph
    print('NATIVE_SPARSE_ACCEPT',flush=True)

if __name__=='__main__':main()
