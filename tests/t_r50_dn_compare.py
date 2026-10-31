"""Native DeltaNet versus the existing fp32-state recurrence."""
import math,time,os
import torch
import torch_npu
from fnx import engine as E
from t_r50_layer_profile import measure


def main():
    torch.npu.set_device(0)
    for T in (8,32,64,128):
        torch.manual_seed(50)
        q=E.l2norm(torch.randn(T,16,128,device='npu:0',dtype=torch.bfloat16))/math.sqrt(128)
        k=E.l2norm(torch.randn_like(q))
        v=torch.randn(T,48,128,device='npu:0',dtype=torch.bfloat16)*.1
        g=-torch.rand(T,48,device='npu:0')*.2
        b=torch.rand(T,48,device='npu:0',dtype=torch.bfloat16)
        s0=torch.randn(1,48,128,128,device='npu:0')*.01
        lens=torch.tensor([T],device='npu:0',dtype=torch.int32)
        qa=q.repeat_interleave(3,dim=1); ka=k.repeat_interleave(3,dim=1)
        s=s0[0]; ref=[]
        for t in range(T):
            s=s*g[t].exp()[:,None,None]
            delta=(v[t]-(s*ka[t].unsqueeze(-1)).sum(-2))*b[t,:,None]
            s=s+ka[t,:, :,None]*delta[:,None,:]
            ref.append((s*qa[t,:,:,None]).sum(-2))
        ref=torch.stack(ref).to(torch.bfloat16)
        state=s0.transpose(-1,-2).to(torch.bfloat16).contiguous()
        fn=lambda:torch_npu.npu_chunk_gated_delta_rule(q,k,v,beta=b,g=g,initial_state=state,actual_seq_lengths=lens,scale=1.)
        got,ss=fn(); torch.npu.synchronize()
        d=(ref.float()-got.float()).abs(); ds=(s-ss.transpose(-1,-2)[0].float()).abs()
        print('DN_NATIVE',T,{'out_rel':float(d.mean()/(ref.float().abs().mean()+1e-9)),
                             'out_max':float(d.max()),'state_rel':float(ds.mean()/(s.abs().mean()+1e-9)),
                             'captured_ms':measure(fn)},flush=True)
        fp=lambda:E.delta_chunk_fp32(qa,ka,v,g,b,s0)
        got,ss=fp(); torch.npu.synchronize()
        d=(ref.float()-got.float()).abs(); ds=(s-ss[0]).abs()
        print('DN_FP32',T,{'out_rel':float(d.mean()/(ref.float().abs().mean()+1e-9)),
                           'out_max':float(d.max()),'state_rel':float(ds.mean()/(s.abs().mean()+1e-9)),
                           'captured_ms':measure(fp)},flush=True)
        os.environ['FNX_DELTA_SOLVER']='matmul'
        got,ss=fp(); torch.npu.synchronize()
        d=(ref.float()-got.float()).abs(); ds=(s-ss[0]).abs()
        print('DN_MATMUL',T,{'out_rel':float(d.mean()/(ref.float().abs().mean()+1e-9)),
                           'state_rel':float(ds.mean()/(s.abs().mean()+1e-9)),
                           'captured_ms':measure(fp)},flush=True)
        os.environ.pop('FNX_DELTA_SOLVER',None)


if __name__=='__main__':
    with torch.no_grad():main()
