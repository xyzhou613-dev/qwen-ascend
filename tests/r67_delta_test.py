"""Bounded preparation must preserve FP32 64-token recurrence exactly."""
import os,time
import torch
from fnx import engine as E

@torch.no_grad()
def main():
    torch.npu.set_device(0)
    window=int(os.environ.get('FNX_DELTA_WINDOW','4096'))
    for T in (8192,16384):
        for correlated in (False,True):
            torch.manual_seed(T+int(correlated))
            q=torch.randn(T,48,128,dtype=torch.bfloat16,device='npu:0')
            k=torch.randn_like(q)
            if correlated:k=k*.1+torch.randn(1,48,128,dtype=k.dtype,device=k.device)
            q=E.l2norm(q)/128**.5;k=E.l2norm(k)
            v=torch.randn_like(q);g=-torch.rand(T,48,device='npu:0')*.1
            beta=torch.rand(T,48,1,dtype=torch.bfloat16,device='npu:0')
            st=torch.randn(1,48,128,128,device='npu:0')*.01
            # Compare against the deployed 8192-token preparation shape.
            # Larger unbounded batches select different CANN matmul kernels.
            os.environ['FNX_DELTA_WINDOW']='0'
            pieces=[];sa=st
            for begin in range(0,T,8192):
                end=min(begin+8192,T)
                part,sa=E.delta_chunk_fp32(q[begin:end],k[begin:end],v[begin:end],
                                         g[begin:end],beta[begin:end],sa)
                pieces.append(part)
            a=torch.cat(pieces,0)
            os.environ['FNX_DELTA_WINDOW']=str(window);b,sb=E.delta_chunk_fp32(q,k,v,g,beta,st)
            rel=((a.float()-b.float()).square().sum()/(a.float().square().sum()+1e-20)).sqrt().item()
            sr=((sa-sb).square().sum()/(sa.square().sum()+1e-20)).sqrt().item()
            print('DELTA_WINDOW_ERROR',T,correlated,rel,sr,'bitexact',torch.equal(a,b) and torch.equal(sa,sb),flush=True)
            assert torch.equal(a,b) and torch.equal(sa,sb),(T,correlated,rel,sr)
            del q,k,v,g,beta,st,a,sa,b,sb
            torch.npu.empty_cache()
    print('DELTA_WINDOW_ACCEPT',flush=True)

if __name__=='__main__':main()
