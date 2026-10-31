"""Correlated keys and beta near one: guard numerical solver stability."""
import os,math
import torch
from fnx import engine as E


def main():
    torch.npu.set_device(1)
    os.environ['FNX_DELTA_SOLVER']=os.environ.get('FNX_DELTA_SOLVER','matmul')
    for T in (64,128,512):
        base=E.l2norm(torch.randn(1,48,128,device='npu:1',dtype=torch.bfloat16))
        k=base.expand(T,-1,-1).contiguous()
        q=k/math.sqrt(128)
        v=torch.randn(T,48,128,device='npu:1',dtype=torch.bfloat16)*.1
        g=torch.full((T,48),-.00001,device='npu:1')
        beta=torch.ones(T,48,device='npu:1',dtype=torch.bfloat16)
        s0=torch.randn(1,48,128,128,device='npu:1')*.01
        s=s0[0];outs=[]
        for i in range(T):
            s=s*g[i].exp()[:,None,None]
            delta=(v[i]-(s*k[i,:,:,None]).sum(-2))*beta[i,:,None]
            s=s+k[i,:,:,None]*delta[:,None,:]
            outs.append((s*q[i,:,:,None]).sum(-2))
        ref=torch.stack(outs).to(torch.bfloat16)
        got,state=E.delta_chunk_fp32(q,k,v,g,beta,s0)
        rel=float((got.float()-ref.float()).abs().mean()/ref.float().abs().mean())
        state_rel=float((s-state[0]).abs().mean()/s.abs().mean())
        print('STRESS',T,rel,state_rel,bool(torch.isfinite(got).all()),flush=True)
        assert torch.isfinite(got).all() and rel<1e-3 and state_rel<1e-3
    print('STRESS_ACCEPT',flush=True)


if __name__=='__main__':
    with torch.no_grad():main()
