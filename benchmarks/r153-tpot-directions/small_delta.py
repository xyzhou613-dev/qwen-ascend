"""Exploratory T<=4 FP32 verifier recurrence with prefix state journals."""
import torch
from fnx import engine as E
from fnx.delta_native import step

original=E.delta_chunk_fp32
journals={}

def small_delta(q,k,v,log_g,beta,state):
    T=q.shape[0]
    if T>4:return original(q,k,v,log_g,beta,state)
    key=(state.data_ptr(),T)
    if key not in journals:
        journals[key]=torch.empty((T+1,)+tuple(state.shape),device=state.device,dtype=torch.float32)
    journal=journals[key];journal[0].copy_(state)
    # The prefill preparation repeated each of 16 q/k heads three times.
    qa=q[:,::3,:].contiguous();ka=k[:,::3,:].contiguous()
    va=v.contiguous();ba=beta.contiguous();decay=log_g.exp().view(T,48,1,1)
    outputs=[]
    for t in range(T):
        outputs.append(step(state,qa[t:t+1],ka[t:t+1],va[t:t+1],ba[t:t+1],
                            decay[t:t+1],compact=True))
        journal[t+1].copy_(state)
    return torch.cat(outputs,0).to(torch.bfloat16),state

@torch.no_grad()
def validate():
    results=[]
    for T in (2,4):
        for seed in range(4):
            torch.manual_seed(15370+seed)
            q=torch.randn(T,16,128,device='npu',dtype=torch.bfloat16)*.03
            k=torch.randn_like(q)*.1
            q=q.repeat_interleave(3,1);k=k.repeat_interleave(3,1)
            v=torch.randn(T,48,128,device='npu',dtype=torch.bfloat16)*.03
            beta=torch.rand(T,48,1,device='npu',dtype=torch.bfloat16)
            log_g=-torch.rand(T,48,device='npu',dtype=torch.float32)*.1
            state=torch.randn(1,48,128,128,device='npu',dtype=torch.float32)*.01
            ref=state.clone();expected=[ref.clone()];outs=[]
            for t in range(T):
                decayed=ref*log_g[t].exp()[None,:,None,None]
                memory=(decayed*k[t].float()[None,:,:,None]).sum(-2)
                delta=(v[t].float()[None]-memory)*beta[t].float()[None]
                ref=decayed+k[t].float()[None,:,:,None]*delta[:,:,None,:]
                outs.append((ref*q[t].float()[None,:,:,None]).sum(-2))
                expected.append(ref.clone())
            actual,last=small_delta(q,k,v,log_g,beta,state)
            journal=journals[(state.data_ptr(),T)]
            err=float((journal-torch.stack(expected)).abs().max())
            output_err=float((actual.float()-torch.cat(outs).to(torch.bfloat16).float()).abs().max())
            assert err<1e-6 and output_err<1e-5,(T,seed,err,output_err)
            results.append(dict(T=T,seed=seed,max_prefix_state_abs=err,max_output_abs=output_err))
            del journals[(state.data_ptr(),T)]
    return results

E.delta_chunk_fp32=small_delta
